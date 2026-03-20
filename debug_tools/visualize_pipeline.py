"""Inspect and debug the data pipeline at each stage."""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


class PipelineInspector:
    """Inspect intermediate data at each stage of the VAD pipeline."""

    def __init__(self, output_dir='logs/pipeline_debug'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def inspect_dataloader_sample(self, data_batch, step=0):
        """Print shapes and stats of a single dataloader batch."""
        print(f"\n{'='*60}")
        print(f"[PipelineInspector] Step {step} - Dataloader Batch Inspection")
        print(f"{'='*60}")

        for key, val in data_batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:30s} | shape: {str(list(val.shape)):20s} | "
                      f"dtype: {val.dtype} | range: [{val.min():.4f}, {val.max():.4f}]")
            elif isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], dict):
                    print(f"  {key:30s} | list[dict] len={len(val)}")
                    # Print first meta dict keys
                    if 'scene_token' in val[0]:
                        print(f"    -> scene_token: {val[0]['scene_token'][:16]}...")
                elif len(val) > 0 and hasattr(val[0], 'tensor'):
                    shapes = [str(list(v.tensor.shape)) for v in val[:3]]
                    print(f"  {key:30s} | list[LiDARInstance3DBoxes] len={len(val)} "
                          f"| first 3 shapes: {shapes}")
                else:
                    print(f"  {key:30s} | list len={len(val)} | type: {type(val[0]).__name__}")
            else:
                print(f"  {key:30s} | type: {type(val).__name__}")
        print()

    def inspect_image_features(self, img_feats, step=0):
        """Inspect extracted image features from backbone+neck."""
        print(f"\n{'='*60}")
        print(f"[PipelineInspector] Step {step} - Image Features")
        print(f"{'='*60}")

        for i, feat in enumerate(img_feats):
            if isinstance(feat, torch.Tensor):
                print(f"  Level {i}: shape={list(feat.shape)} | "
                      f"mean={feat.mean():.6f} | std={feat.std():.6f} | "
                      f"norm={feat.norm():.4f}")
            else:
                print(f"  Level {i}: type={type(feat).__name__}")
        print()

    def inspect_bev_embedding(self, bev_embed, step=0):
        """Inspect BEV embeddings from encoder output."""
        print(f"\n{'='*60}")
        print(f"[PipelineInspector] Step {step} - BEV Embedding")
        print(f"{'='*60}")

        if isinstance(bev_embed, torch.Tensor):
            print(f"  Shape: {list(bev_embed.shape)}")
            print(f"  Dtype: {bev_embed.dtype}")
            print(f"  Mean: {bev_embed.mean():.6f}")
            print(f"  Std:  {bev_embed.std():.6f}")
            print(f"  Norm: {bev_embed.norm():.4f}")
            print(f"  NaN count: {bev_embed.isnan().sum().item()}")
            print(f"  Inf count: {bev_embed.isinf().sum().item()}")
        else:
            print(f"  Type: {type(bev_embed).__name__}")
        print()

    def inspect_head_outputs(self, outs, step=0):
        """Inspect VADHead outputs (detection, map, planning)."""
        print(f"\n{'='*60}")
        print(f"[PipelineInspector] Step {step} - Head Outputs")
        print(f"{'='*60}")

        for key, val in outs.items():
            if isinstance(val, list):
                for layer_i, layer_val in enumerate(val):
                    if isinstance(layer_val, dict):
                        for sub_key, sub_val in layer_val.items():
                            if isinstance(sub_val, torch.Tensor):
                                print(f"  {key}[{layer_i}].{sub_key:20s} | "
                                      f"shape={list(sub_val.shape)} | "
                                      f"range=[{sub_val.min():.4f}, {sub_val.max():.4f}]")
            elif isinstance(val, torch.Tensor):
                print(f"  {key:30s} | shape={list(val.shape)} | "
                      f"range=[{val.min():.4f}, {val.max():.4f}]")
            elif isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, torch.Tensor):
                        print(f"  {key}.{sub_key:20s} | shape={list(sub_val.shape)}")
        print()

    def inspect_losses(self, losses, step=0):
        """Print detailed loss breakdown."""
        print(f"\n{'='*60}")
        print(f"[PipelineInspector] Step {step} - Loss Breakdown")
        print(f"{'='*60}")

        total = 0.0
        loss_items = []
        for key, val in losses.items():
            if isinstance(val, torch.Tensor):
                v = val.item()
                loss_items.append((key, v))
                if key != 'loss':
                    total += v

        # Sort by magnitude
        loss_items.sort(key=lambda x: -abs(x[1]))

        for key, val in loss_items:
            bar_len = int(val / max(x[1] for x in loss_items) * 30) if loss_items else 0
            bar = '#' * bar_len
            print(f"  {key:30s} | {val:10.6f} | {bar}")

        print(f"  {'-'*50}")
        print(f"  {'Computed Total':30s} | {total:10.6f}")
        print()

    def save_feature_snapshot(self, name, tensor, step=0):
        """Save a tensor snapshot to disk for later analysis."""
        if isinstance(tensor, torch.Tensor):
            path = os.path.join(self.output_dir, f"step{step}_{name}.pt")
            torch.save(tensor.cpu(), path)
            print(f"[PipelineInspector] Saved {name} to {path}")

    def inspect_predictions(self, bbox_results, step=0):
        """Inspect final prediction outputs."""
        print(f"\n{'='*60}")
        print(f"[PipelineInspector] Step {step} - Predictions")
        print(f"{'='*60}")

        if 'boxes_3d' in bbox_results:
            boxes = bbox_results['boxes_3d']
            n = len(boxes) if hasattr(boxes, '__len__') else 0
            print(f"  Detected boxes: {n}")

            if 'scores_3d' in bbox_results:
                scores = bbox_results['scores_3d']
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    print(f"  Score > 0.3: {(scores > 0.3).sum()}")
                    print(f"  Score > 0.5: {(scores > 0.5).sum()}")

        if 'map_pts_3d' in bbox_results:
            map_pts = bbox_results['map_pts_3d']
            if isinstance(map_pts, torch.Tensor):
                print(f"  Map elements: {map_pts.shape[0]}")

        if 'ego_fut_preds' in bbox_results:
            ego_preds = bbox_results['ego_fut_preds']
            if isinstance(ego_preds, torch.Tensor):
                print(f"  Ego planning shape: {list(ego_preds.shape)}")
                # Show trajectory end points
                traj_cumsum = ego_preds.cumsum(dim=-2)
                print(f"  Trajectory end points (cumsum):")
                for cmd_idx in range(ego_preds.shape[0]):
                    end = traj_cumsum[cmd_idx, -1]
                    print(f"    cmd={cmd_idx}: [{end[0]:.2f}, {end[1]:.2f}] m")

        print()
