"""Analyze training losses and gradients for debugging convergence issues."""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch


class LossAnalyzer:
    """Analyze loss trends, gradient flow, and convergence from training logs."""

    def __init__(self, work_dir='work_dirs'):
        self.work_dir = work_dir

    def parse_log_file(self, log_path):
        """Parse mmdet training log into structured data."""
        records = []
        current_epoch = 0

        with open(log_path, 'r') as f:
            for line in f:
                # Match iteration log lines
                iter_match = re.search(r'Epoch\s*\[(\d+)\]\[(\d+)/\d+\]', line)
                if iter_match:
                    current_epoch = int(iter_match.group(1))
                    step = int(iter_match.group(2))

                    # Extract all key=value pairs
                    kv_pattern = re.compile(r'(\w[\w_]*)\s*[:=]\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)')
                    kvs = kv_pattern.findall(line)

                    record = {'epoch': current_epoch, 'step': step}
                    for k, v in kvs:
                        try:
                            record[k] = float(v)
                        except ValueError:
                            pass
                    if len(record) > 2:
                        records.append(record)

        return records

    def plot_loss_curves(self, log_path, save_path=None):
        """Plot all loss components from a training log."""
        records = self.parse_log_file(log_path)
        if not records:
            print(f"[LossAnalyzer] No records found in {log_path}")
            return

        # Collect all loss keys
        loss_keys = set()
        for r in records:
            for k in r:
                if k.startswith('loss') and k not in ('epoch', 'step'):
                    loss_keys.add(k)
        loss_keys = sorted(loss_keys)

        # Extract values
        steps = [r['step'] + r['epoch'] * 1000 for r in records]  # approximate global step
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Individual losses
        ax1 = axes[0]
        for key in loss_keys:
            vals = [r.get(key, np.nan) for r in records]
            ax1.plot(steps, vals, label=key, alpha=0.7, linewidth=0.8)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Individual Loss Components')
        ax1.legend(fontsize=6, ncol=3, loc='upper right')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Total loss
        ax2 = axes[1]
        total_loss_key = 'loss'
        if total_loss_key in loss_keys:
            vals = [r.get(total_loss_key, np.nan) for r in records]
            ax2.plot(steps, vals, color='#e74c3c', linewidth=1.5, label='Total Loss')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Total Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Sum all losses
            summed = []
            for r in records:
                total = sum(r.get(k, 0) for k in loss_keys)
                summed.append(total)
            ax2.plot(steps, summed, color='#e74c3c', linewidth=1.5, label='Sum of Losses')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Sum of All Losses')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[LossAnalyzer] Saved loss curves to {save_path}")
        else:
            plt.show()
        return fig

    def check_gradient_flow(self, model, save_path=None):
        """Visualize gradient magnitudes across model layers."""
        ave_grads = []
        max_grads = []
        layers = []
        has_grad = []

        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
                has_grad.append(True)
            elif param.requires_grad:
                layers.append(name)
                ave_grads.append(0)
                max_grads.append(0)
                has_grad.append(False)

        if not layers:
            print("[LossAnalyzer] No gradients found. Has backward() been called?")
            return

        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.15), 6))
        x = range(len(layers))

        bars = ax.bar(x, max_grads, alpha=0.3, lw=1, color='c', label='Max Grad')
        ax.bar(x, ave_grads, alpha=0.7, lw=1, color='b', label='Mean Grad')

        # Highlight layers with zero gradients
        zero_grad_x = [i for i, g in enumerate(has_grad) if not g]
        if zero_grad_x:
            ax.bar(zero_grad_x, [0.001] * len(zero_grad_x), alpha=0.5, color='r',
                   label='No Gradient')

        ax.set_xlabel('Layers')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Gradient Flow')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([l.split('.')[-1][:12] for l in layers], rotation=90, fontsize=5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[LossAnalyzer] Saved gradient flow to {save_path}")
        else:
            plt.show()

        # Print summary
        print(f"[LossAnalyzer] Gradient Summary:")
        print(f"  Layers with gradients: {sum(has_grad)}/{len(layers)}")
        print(f"  Layers without gradients: {len(layers) - sum(has_grad)}")
        if ave_grads:
            nonzero = [g for g in ave_grads if g > 0]
            if nonzero:
                print(f"  Mean grad range: [{min(nonzero):.2e}, {max(nonzero):.2e}]")

        return fig

    def check_weight_distribution(self, model, save_path=None):
        """Visualize weight distributions for key layers."""
        weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                # Sample key layers
                parts = name.split('.')
                short_name = '.'.join(parts[-3:]) if len(parts) > 3 else name
                if any(k in name for k in ['backbone', 'neck', 'transformer', 'head']):
                    weights[short_name] = param.data.cpu().numpy().flatten()

        if not weights:
            print("[LossAnalyzer] No weights found for visualization")
            return

        n_plots = min(len(weights), 16)
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()

        for i, (name, w) in enumerate(list(weights.items())[:n_plots]):
            axes[i].hist(w, bins=50, alpha=0.7, color='#3498db')
            axes[i].set_title(name[:25], fontsize=7)
            axes[i].tick_params(labelsize=6)
            axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)

        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Weight Distributions', fontsize=14)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[LossAnalyzer] Saved weight distributions to {save_path}")
        else:
            plt.show()
        return fig
