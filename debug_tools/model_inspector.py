"""Inspect model architecture, parameter counts, and intermediate outputs."""
import torch
import torch.nn as nn


class ModelInspector:
    """Inspect VAD model architecture and parameters."""

    def __init__(self, model):
        self.model = model

    def summary(self):
        """Print model architecture summary with parameter counts."""
        print("\n" + "=" * 70)
        print("VAD Model Architecture Summary")
        print("=" * 70)

        total_params = 0
        trainable_params = 0
        module_params = {}

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules
                n_params = sum(p.numel() for p in module.parameters())
                n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if n_params > 0:
                    module_params[name] = (n_params, n_trainable)
                    total_params += n_params
                    trainable_params += n_trainable

        # Group by top-level component
        groups = {}
        for name, (total, trainable) in module_params.items():
            top = name.split('.')[0] if '.' in name else name
            if top not in groups:
                groups[top] = {'total': 0, 'trainable': 0, 'modules': 0}
            groups[top]['total'] += total
            groups[top]['trainable'] += trainable
            groups[top]['modules'] += 1

        print(f"\n{'Component':<30s} {'Params':>12s} {'Trainable':>12s} {'Modules':>8s}")
        print("-" * 70)
        for group, info in sorted(groups.items(), key=lambda x: -x[1]['total']):
            print(f"  {group:<28s} {info['total']:>12,} {info['trainable']:>12,} {info['modules']:>8,}")

        print("-" * 70)
        print(f"  {'TOTAL':<28s} {total_params:>12,} {trainable_params:>12,}")
        print(f"  {'Trainable ratio':<28s} {trainable_params/total_params*100:>11.1f}%")
        print()

    def check_forward_hooks(self):
        """Register hooks to capture intermediate outputs."""
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'shape': list(output.shape),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'has_nan': output.isnan().any().item(),
                        'has_inf': output.isinf().any().item(),
                    }
                elif isinstance(output, (list, tuple)):
                    for i, o in enumerate(output):
                        if isinstance(o, torch.Tensor):
                            activations[f'{name}[{i}]'] = {
                                'shape': list(o.shape),
                                'mean': o.mean().item(),
                                'std': o.std().item(),
                                'has_nan': o.isnan().any().item(),
                                'has_inf': o.isinf().any().item(),
                            }
            return hook

        handles = []
        for name, module in self.model.named_modules():
            if any(key in name for key in ['backbone', 'neck', 'transformer', 'head']):
                if len(list(module.children())) == 0:
                    h = module.register_forward_hook(hook_fn(name))
                    handles.append(h)

        return activations, handles

    def check_device_dtype(self):
        """Check model device and dtype consistency."""
        print("\n" + "=" * 70)
        print("Device & Dtype Check")
        print("=" * 70)

        devices = set()
        dtypes = set()
        issues = []

        for name, param in self.model.named_parameters():
            devices.add(param.device)
            dtypes.add(param.dtype)

        print(f"  Devices: {devices}")
        print(f"  Dtypes:  {dtypes}")

        if len(devices) > 1:
            issues.append(f"Multiple devices detected: {devices}")
        if len(dtypes) > 1:
            issues.append(f"Multiple dtypes detected: {dtypes}")

        # Check for specific components
        if hasattr(self.model, 'img_backbone'):
            bb_params = list(self.model.img_backbone.parameters())
            if bb_params:
                print(f"  Backbone device: {bb_params[0].device}, dtype: {bb_params[0].dtype}")

        if hasattr(self.model, 'pts_bbox_head'):
            head_params = list(self.model.pts_bbox_head.parameters())
            if head_params:
                print(f"  Head device: {head_params[0].device}, dtype: {head_params[0].dtype}")

        if issues:
            print("\n  WARNING - Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("\n  OK - All parameters on same device/dtype")
        print()

    def measure_inference_time(self, dummy_input, n_runs=10, warmup=3):
        """Measure inference time for profiling."""
        import time

        print("\n" + "=" * 70)
        print("Inference Time Profiling")
        print("=" * 70)

        self.model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model.extract_feat(**dummy_input)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                _ = self.model.extract_feat(**dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start)

        times = sorted(times)
        print(f"  Runs: {n_runs}")
        print(f"  Mean: {sum(times)/len(times)*1000:.1f} ms")
        print(f"  Median: {times[len(times)//2]*1000:.1f} ms")
        print(f"  Min: {times[0]*1000:.1f} ms")
        print(f"  Max: {times[-1]*1000:.1f} ms")
        print()

    def find_unfreeze_candidates(self, freeze_keywords=None):
        """Find which layers to unfreeze for fine-tuning."""
        if freeze_keywords is None:
            freeze_keywords = ['backbone', 'bn', 'norm']

        print("\n" + "=" * 70)
        print("Layer Freeze/Unfreeze Analysis")
        print("=" * 70)

        frozen = []
        trainable = []

        for name, param in self.model.named_parameters():
            is_frozen = any(kw in name for kw in freeze_keywords)
            if is_frozen:
                frozen.append((name, param.shape, param.numel()))
            else:
                trainable.append((name, param.shape, param.numel()))

        frozen_params = sum(p[2] for p in frozen)
        trainable_params = sum(p[2] for p in trainable)
        total = frozen_params + trainable_params

        print(f"  Frozen layers ({len(frozen)}): {frozen_params:,} params ({frozen_params/total*100:.1f}%)")
        print(f"  Trainable layers ({len(trainable)}): {trainable_params:,} params ({trainable_params/total*100:.1f}%)")
        print()
