"""
Debug a single sample through the full VAD pipeline.
Run from project root: python scripts/debug_one_sample.py

This script:
1. Loads config and builds model
2. Loads one sample from dataloader
3. Runs forward pass with detailed inspection at each stage
4. Visualizes BEV, predictions, and losses
"""
import sys
import os
import argparse
import copy

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'VAD'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from mmcv import Config
from mmcv.utils import import_modules_from_strings

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from debug_tools import BEVVisualizer, PipelineInspector, LossAnalyzer, ModelInspector


def parse_args():
    parser = argparse.ArgumentParser(description='Debug single sample through VAD pipeline')
    parser.add_argument('config', default='configs/VAD_base_debug.py', help='config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file (optional)')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id')
    parser.add_argument('--output-dir', default='logs/debug_one_sample', help='output directory')
    parser.add_argument('--sample-idx', type=int, default=0, help='which sample to debug')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg = Config.fromfile(args.config)

    # Import plugin
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = cfg.plugin_dir
        module_path = plugin_dir.replace('/', '.').rstrip('.')
        importlib.import_module(module_path)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========== Step 1: Build Dataset ==========
    print("\n" + "=" * 60)
    print("STEP 1: Building dataset...")
    print("=" * 60)

    dataset = build_dataset(cfg.data.train)
    print(f"Dataset type: {type(dataset).__name__}")
    print(f"Dataset length: {len(dataset)}")

    # Get one sample
    sample = dataset[args.sample_idx]
    inspector = PipelineInspector(output_dir=args.output_dir)
    inspector.inspect_dataloader_sample(sample, step=0)

    # ========== Step 2: Build Model ==========
    print("\n" + "=" * 60)
    print("STEP 2: Building model...")
    print("=" * 60)

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)

    model = model.to(device)
    model.eval()

    # Inspect model
    model_insp = ModelInspector(model)
    model_insp.summary()
    model_insp.check_device_dtype()

    # ========== Step 3: Prepare Batch ==========
    print("\n" + "=" * 60)
    print("STEP 3: Preparing batch...")
    print("=" * 60)

    # Use mmdet's collate to prepare a batch
    from mmcv.parallel import collate, scatter
    from mmdet.datasets.builder import build_dataloader

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        seed=0,
    )

    data_batch = next(iter(dataloader))
    data_batch = scatter(data_batch, [device])[0]

    print("Batch keys:", list(data_batch.keys()))
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")

    # ========== Step 4: Forward Pass ==========
    print("\n" + "=" * 60)
    print("STEP 4: Running forward pass...")
    print("=" * 60)

    # Register hooks for intermediate inspection
    activations, hook_handles = model_insp.check_forward_hooks()

    with torch.no_grad():
        # Extract image features
        img = data_batch.get('img')
        img_metas = data_batch.get('img_metas')

        if img is not None:
            print(f"\nInput image shape: {list(img.shape)}")

            # Handle queue dimension for temporal
            if img.dim() == 6:  # [B, T, N, C, H, W]
                bs, T, N, C, H, W = img.shape
                print(f"  Temporal queue: T={T}, Cams={N}")
                img_current = img[:, -1, ...]  # current frame
            else:
                img_current = img

            # Extract features
            img_feats = model.extract_feat(img=img_current.reshape(-1, *img_current.shape[-3:]),
                                           img_metas=img_metas)
            inspector.inspect_image_features(img_feats, step=1)

            # Visualize one feature level
            if img_feats:
                bev_vis = BEVVisualizer()
                feat_to_vis = img_feats[0] if isinstance(img_feats[0], torch.Tensor) else img_feats[0]
                bev_vis.visualize_bev_feature(
                    feat_to_vis,
                    save_path=os.path.join(args.output_dir, 'feature_heatmap.png')
                )

    # Print hook captured activations
    print("\nCaptured intermediate activations:")
    for name, info in sorted(activations.items()):
        nan_flag = " [NaN!]" if info['has_nan'] else ""
        inf_flag = " [Inf!]" if info['has_inf'] else ""
        print(f"  {name:50s} | shape={info['shape']} | "
              f"mean={info['mean']:.6f} | std={info['std']:.6f}{nan_flag}{inf_flag}")

    # Remove hooks
    for h in hook_handles:
        h.remove()

    # ========== Step 5: Visualize ==========
    print("\n" + "=" * 60)
    print("STEP 5: Visualization...")
    print("=" * 60)

    bev_vis = BEVVisualizer()

    # Visualize GT if available
    gt_bboxes = data_batch.get('gt_bboxes_3d')
    gt_labels = data_batch.get('gt_labels_3d')

    if gt_bboxes is not None and gt_labels is not None:
        img_meta_sample = img_metas[0][0] if isinstance(img_metas[0], list) else img_metas[0]
        bev_vis.visualize_sample(
            img_meta_sample,
            gt_bboxes_3d=gt_bboxes[0] if isinstance(gt_bboxes, list) else gt_bboxes,
            gt_labels_3d=gt_labels[0] if isinstance(gt_labels, list) else gt_labels,
            save_path=os.path.join(args.output_dir, 'gt_visualization.png')
        )

    # ========== Done ==========
    print("\n" + "=" * 60)
    print("Debug complete!")
    print(f"Outputs saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
