"""
Training script with built-in debug hooks.
Run from project root: python scripts/train_debug.py configs/VAD_base_debug.py

This wraps VAD training with:
- Gradient monitoring
- Loss tracking
- Periodic visualization
- NaN/Inf detection
"""
import sys
import os
import argparse
import copy
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'VAD'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

import cv2
cv2.setNumThreads(1)

from debug_tools import LossAnalyzer, ModelInspector


class DebugHooks:
    """Training debug hooks for NaN detection and gradient monitoring."""

    def __init__(self, model, output_dir='logs/debug_hooks', check_interval=50):
        self.model = model
        self.output_dir = output_dir
        self.check_interval = check_interval
        self.loss_analyzer = LossAnalyzer(output_dir)
        self.step = 0
        self.nan_detected = False
        os.makedirs(output_dir, exist_ok=True)

        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks on key layers for gradient monitoring."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(self._grad_hook(name, param))

    def _grad_hook(self, name, param):
        def hook(grad):
            self.step += 1
            if self.step % self.check_interval == 0:
                # Check for NaN/Inf
                if grad.isnan().any():
                    print(f"[NaN DETECTED] Step {self.step}, Layer: {name}")
                    self.nan_detected = True
                if grad.isinf().any():
                    print(f"[Inf DETECTED] Step {self.step}, Layer: {name}")
                    self.nan_detected = True

                # Log gradient stats
                if self.step % (self.check_interval * 10) == 0:
                    print(f"[Grad Stats] Step {self.step}, {name}: "
                          f"mean={grad.mean():.2e}, std={grad.std():.2e}, "
                          f"max={grad.abs().max():.2e}")
            return grad
        return hook

    def on_train_begin(self):
        """Called at the start of training."""
        print("\n[DebugHooks] Training started")
        model_insp = ModelInspector(self.model)
        model_insp.summary()
        model_insp.check_device_dtype()

    def on_train_end(self):
        """Called at end of training."""
        print("\n[DebugHooks] Training ended")
        self.loss_analyzer.check_gradient_flow(
            self.model,
            save_path=os.path.join(self.output_dir, 'final_gradient_flow.png')
        )
        self.loss_analyzer.check_weight_distribution(
            self.model,
            save_path=os.path.join(self.output_dir, 'weight_distributions.png')
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Train VAD with debug hooks')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug-interval', type=int, default=50,
                        help='check gradients every N steps')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Import plugin
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = cfg.plugin_dir
        module_path = plugin_dir.replace('/', '.').rstrip('.')
        importlib.import_module(module_path)

    # Work dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    os.makedirs(cfg.work_dir, exist_ok=True)

    # Setup logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name='mmdet')

    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(f'Debug interval: {args.debug_interval}')

    # Set seed
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)
    cfg.seed = args.seed

    # Build model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    logger.info(f'Model:\n{model}')

    # Setup debug hooks
    debug_hooks = DebugHooks(
        model,
        output_dir=os.path.join(cfg.work_dir, 'debug'),
        check_interval=args.debug_interval
    )
    debug_hooks.on_train_begin()

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]

    # Build dataloader
    from mmdet.datasets.builder import build_dataloader
    dataloader = build_dataloader(
        datasets[0],
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        seed=args.seed,
    )

    # Simple training loop with debug
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer['lr'],
        weight_decay=cfg.optimizer.get('weight_decay', 0.01)
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    for epoch in range(cfg.runner.max_epochs):
        logger.info(f'Epoch {epoch}/{cfg.runner.max_epochs}')

        for step, data_batch in enumerate(dataloader):
            # Move to device
            data_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in data_batch.items()}

            optimizer.zero_grad()
            outputs = model(return_loss=True, **data_batch)

            total_loss = outputs['loss'] if 'loss' in outputs else sum(
                v for k, v in outputs.items() if 'loss' in k and isinstance(v, torch.Tensor)
            )

            total_loss.backward()

            # Check gradients before step
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer_config.get('grad_clip', {}).get('max_norm', 35))

            optimizer.step()

            if step % cfg.log_config.interval == 0:
                loss_str = ' | '.join(f'{k}: {v.item():.4f}' for k, v in outputs.items()
                                      if isinstance(v, torch.Tensor) and 'loss' in k)
                logger.info(f'Epoch [{epoch}][{step}/{len(dataloader)}] {loss_str}')

    debug_hooks.on_train_end()
    logger.info(f'Debug training complete. Outputs in: {cfg.work_dir}')


if __name__ == '__main__':
    main()
