"""BEV feature and prediction visualization for debugging."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import torch


class BEVVisualizer:
    """Visualize BEV features, detections, map elements, and planning trajectories."""

    # nuScenes colormap for 10 classes
    CLASS_COLORS = {
        'car': '#e74c3c',
        'truck': '#e67e22',
        'construction_vehicle': '#f39c12',
        'bus': '#d35400',
        'trailer': '#c0392b',
        'barrier': '#7f8c8d',
        'motorcycle': '#2ecc71',
        'bicycle': '#27ae60',
        'pedestrian': '#3498db',
        'traffic_cone': '#9b59b6',
    }

    MAP_COLORS = {
        'divider': '#e74c3c',
        'ped_crossing': '#3498db',
        'boundary': '#2ecc71',
    }

    def __init__(self, pc_range=None, bev_h=200, bev_w=200):
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = (self.pc_range[0], self.pc_range[3])
        self.y_range = (self.pc_range[1], self.pc_range[4])

    def _bev_to_world(self, bev_x, bev_y):
        """Convert BEV grid coords to world coords."""
        world_x = bev_x / self.bev_w * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        world_y = bev_y / self.bev_h * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return world_x, world_y

    def visualize_sample(self, img_metas, bbox_results=None, gt_bboxes_3d=None,
                         gt_labels_3d=None, save_path=None, figsize=(16, 16)):
        """Visualize a single sample with predictions and/or GT."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.tick_params(colors='white')

        # Draw ego vehicle at origin
        ego_rect = mpatches.FancyBboxPatch(
            (-1.0, -0.5), 2.0, 4.0,
            boxstyle="round,pad=0.1",
            facecolor='#f1c40f', edgecolor='white', linewidth=2, alpha=0.8
        )
        ax.add_patch(ego_rect)
        ax.text(0, 2.5, 'EGO', ha='center', va='center', color='white', fontsize=8, fontweight='bold')

        # Draw GT boxes
        if gt_bboxes_3d is not None and gt_labels_3d is not None:
            self._draw_boxes(ax, gt_bboxes_3d, gt_labels_3d, is_gt=True)

        # Draw predicted boxes
        if bbox_results is not None:
            if 'boxes_3d' in bbox_results:
                pred_boxes = bbox_results['boxes_3d']
                pred_labels = bbox_results.get('labels_3d', None)
                scores = bbox_results.get('scores_3d', None)
                self._draw_boxes(ax, pred_boxes, pred_labels, scores=scores, is_gt=False)

            # Draw predicted map elements
            if 'map_pts_3d' in bbox_results:
                self._draw_map_elements(ax, bbox_results)

            # Draw ego planned trajectory
            if 'ego_fut_preds' in bbox_results:
                self._draw_planning(ax, bbox_results['ego_fut_preds'])

            # Draw agent predicted trajectories
            if 'trajs_3d' in bbox_results and 'boxes_3d' in bbox_results:
                self._draw_agent_trajs(ax, bbox_results)

        # Title
        scene_token = img_metas.get('scene_token', 'N/A') if isinstance(img_metas, dict) else 'N/A'
        sample_idx = img_metas.get('sample_idx', 'N/A') if isinstance(img_metas, dict) else 'N/A'
        ax.set_title(f'Scene: {str(scene_token)[:8]}... | Sample: {sample_idx}',
                     color='white', fontsize=12)

        # Legend
        legend_patches = [
            mpatches.Patch(color='#f1c40f', label='Ego'),
            mpatches.Patch(color='#e74c3c', alpha=0.6, label='GT (vehicle)'),
            mpatches.Patch(color='#3498db', alpha=0.6, label='GT (pedestrian)'),
            mpatches.Patch(color='#2ecc71', alpha=0.5, label='Pred (score>0.3)'),
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
                  facecolor='#2d2d44', edgecolor='white', labelcolor='white')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
            print(f"[BEVVisualizer] Saved to {save_path}")
        else:
            plt.show()
        return fig

    def _draw_boxes(self, ax, boxes, labels, scores=None, is_gt=True):
        """Draw 3D bounding boxes in BEV."""
        if hasattr(boxes, 'tensor'):
            boxes_np = boxes.tensor.cpu().numpy()
        elif isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = np.array(boxes)

        class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

        for i in range(len(boxes_np)):
            x, y, z, dx, dy, dz, yaw = boxes_np[i, :7]
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            # Box corners in BEV
            corners = np.array([
                [-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2]
            ])
            rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            corners = corners @ rot.T
            corners[:, 0] += x
            corners[:, 1] += y

            cls_name = class_names[labels[i]] if labels is not None and i < len(labels) else 'unknown'
            if is_gt:
                color = self.CLASS_COLORS.get(cls_name, '#ffffff')
                alpha = 0.6
                linewidth = 1.5
                linestyle = '-'
            else:
                color = self.CLASS_COLORS.get(cls_name, '#ffffff')
                alpha = 0.4
                linewidth = 1.0
                linestyle = '--'

            if scores is not None and not is_gt:
                score = scores[i].item() if hasattr(scores[i], 'item') else scores[i]
                if score < 0.3:
                    continue

            poly = plt.Polygon(corners, fill=True, facecolor=color, edgecolor=color,
                              alpha=alpha, linewidth=linewidth, linestyle=linestyle)
            ax.add_patch(poly)

            # Direction indicator
            front = (corners[0] + corners[1]) / 2
            center = np.array([x, y])
            ax.annotate('', xy=front, xytext=center,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    def _draw_map_elements(self, ax, bbox_results):
        """Draw map elements (dividers, crossings, boundaries)."""
        map_pts = bbox_results.get('map_pts_3d', None)
        map_labels = bbox_results.get('map_labels_3d', None)
        map_scores = bbox_results.get('map_scores_3d', None)

        if map_pts is None:
            return

        if isinstance(map_pts, torch.Tensor):
            map_pts = map_pts.cpu().numpy()
        if map_labels is not None and isinstance(map_labels, torch.Tensor):
            map_labels = map_labels.cpu().numpy()

        map_class_names = ['divider', 'ped_crossing', 'boundary']

        for i in range(len(map_pts)):
            if map_scores is not None:
                score = map_scores[i].item() if hasattr(map_scores[i], 'item') else map_scores[i]
                if score < 0.3:
                    continue

            pts = map_pts[i]
            cls_name = map_class_names[map_labels[i]] if map_labels is not None and i < len(map_labels) else 'divider'
            color = self.MAP_COLORS.get(cls_name, '#ffffff')

            if isinstance(pts, torch.Tensor):
                pts = pts.cpu().numpy()

            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.7)

    def _draw_planning(self, ax, ego_fut_preds):
        """Draw ego planned trajectory."""
        if isinstance(ego_fut_preds, torch.Tensor):
            ego_fut_preds = ego_fut_preds.cpu().numpy()

        # ego_fut_preds shape: [num_cmds, fut_ts, 2] - incremental
        # Take the first command mode for visualization
        if ego_fut_preds.ndim == 3:
            traj = ego_fut_preds[0]  # [fut_ts, 2]
        else:
            traj = ego_fut_preds

        # Convert incremental to absolute
        traj_abs = np.cumsum(traj, axis=0)
        traj_abs = np.concatenate([np.zeros((1, 2)), traj_abs], axis=0)

        ax.plot(traj_abs[:, 0], traj_abs[:, 1], 'w-', linewidth=3, alpha=0.9, label='Planned Path')
        ax.scatter(traj_abs[-1, 0], traj_abs[-1, 1], c='white', s=80, zorder=5, marker='*')

        # Time annotations
        for t in range(1, len(traj_abs)):
            ax.text(traj_abs[t, 0], traj_abs[t, 1] + 0.5, f'{t}s',
                    ha='center', va='bottom', color='white', fontsize=7, alpha=0.7)

    def _draw_agent_trajs(self, ax, bbox_results, fut_mode=6, fut_ts=6):
        """Draw agent predicted trajectories."""
        trajs = bbox_results['trajs_3d']
        boxes = bbox_results['boxes_3d']
        scores = bbox_results.get('scores_3d', None)

        if isinstance(trajs, torch.Tensor):
            trajs = trajs.cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes.tensor.cpu().numpy()
        else:
            boxes_np = boxes.tensor.cpu().numpy() if hasattr(boxes, 'tensor') else np.array(boxes)

        for i in range(min(len(trajs), 20)):  # limit to 20 agents
            if scores is not None:
                s = scores[i].item() if hasattr(scores[i], 'item') else scores[i]
                if s < 0.4:
                    continue

            agent_traj = trajs[i].reshape(fut_mode, fut_ts, 2)  # [mode, ts, 2]
            center = boxes_np[i, :2]

            # Draw best mode
            best_mode = agent_traj[0]  # first mode
            traj_abs = np.cumsum(best_mode, axis=0) + center
            traj_abs = np.concatenate([center.reshape(1, 2), traj_abs], axis=0)

            ax.plot(traj_abs[:, 0], traj_abs[:, 1], color='#95a5a6',
                    linewidth=1, alpha=0.4, linestyle='--')

    def visualize_bev_feature(self, bev_feat, save_path=None, figsize=(12, 12)):
        """Visualize BEV feature map as heatmap."""
        if isinstance(bev_feat, torch.Tensor):
            bev_feat = bev_feat.cpu().numpy()

        # Average across channels
        if bev_feat.ndim == 3:  # [C, H, W]
            bev_map = np.mean(np.abs(bev_feat), axis=0)
        elif bev_feat.ndim == 4:  # [B, C, H, W]
            bev_map = np.mean(np.abs(bev_feat[0]), axis=0)
        else:
            print(f"[BEVVisualizer] Unexpected BEV feature shape: {bev_feat.shape}")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(bev_map, cmap='hot', aspect='equal',
                       extent=[self.x_range[0], self.x_range[1],
                               self.y_range[0], self.y_range[1]],
                       origin='lower')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('BEV Feature Map (channel avg)')
        plt.colorbar(im, ax=ax, label='Feature Magnitude')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[BEVVisualizer] BEV feature saved to {save_path}")
        else:
            plt.show()
        return fig

    def visualize_loss_curve(self, log_file, save_path=None):
        """Parse training log and plot loss curves."""
        losses = {}
        epochs = []

        with open(log_file, 'r') as f:
            for line in f:
                if 'epoch' in line and 'loss' in line:
                    # Simple parsing - adjust based on actual log format
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part.startswith('loss') and i + 2 < len(parts):
                            try:
                                val = float(parts[i + 2].rstrip(','))
                                losses.setdefault(part, []).append(val)
                            except (ValueError, IndexError):
                                pass

        if not losses:
            print(f"[BEVVisualizer] No loss data found in {log_file}")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        for name, values in losses.items():
            ax.plot(values, label=name, alpha=0.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend(fontsize=7, ncol=3)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[BEVVisualizer] Loss curve saved to {save_path}")
        else:
            plt.show()
        return fig
