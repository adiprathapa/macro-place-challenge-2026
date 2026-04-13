"""
Force-directed placement optimizer with iterative spreading.

Directly optimizes macro positions using Adam with the same proven loss functions
as the GNN approach (wirelength + density + overlap + spreading), but without
neural network overhead. This enables 4-5x more optimization iterations in the
same time budget, producing better convergence.

The optimizer also supports an optional second phase with finer-grid density
refinement for improved spreading control.
"""

import torch
import math
import time

from losses import wirelength_loss, overlap_loss, density_loss, spreading_loss, total_loss


class ElectrostaticOptimizer:
    """
    Direct position optimizer with iterative density spreading.

    Uses Adam on position variables with the same loss functions as the GNN,
    but bypasses the neural network for direct gradient computation.
    Optionally adds a fine-grid density refinement phase.
    """

    def __init__(self, benchmark, graph, device, grid_size=32):
        self.benchmark = benchmark
        self.graph = graph
        self.device = device
        self.grid_size = grid_size

        self.canvas_w = benchmark.canvas_width
        self.canvas_h = benchmark.canvas_height
        self.num_hard = benchmark.num_hard_macros
        self.num_macros = benchmark.num_macros
        self.fixed_mask = benchmark.macro_fixed.to(device)
        self.movable_mask = ~self.fixed_mask
        self.sizes = benchmark.macro_sizes.to(device)

    def _clamp(self, positions, fixed_pos):
        """Hard clamp to canvas bounds and enforce fixed positions."""
        hw = self.sizes[:, 0] / 2
        hh = self.sizes[:, 1] / 2
        positions[:, 0] = torch.max(torch.min(positions[:, 0], self.canvas_w - hw), hw)
        positions[:, 1] = torch.max(torch.min(positions[:, 1], self.canvas_h - hh), hh)
        positions[self.fixed_mask] = fixed_pos[self.fixed_mask]
        return positions

    def optimize(self, init_positions, num_iters=800, time_limit=600.0):
        """
        Two-phase Adam optimization on macro positions.

        Phase 1 (80%): Use the proven total_loss (same as GNN training) but
        optimize positions directly. This gives the same quality of loss
        gradients but with 4-5x more iterations than the GNN approach.

        Phase 2 (20%): Fine-grid density + strong overlap penalty to prepare
        for legalization.

        Args:
            init_positions: [num_macros, 2] from GNN initialization
            num_iters: max iterations
            time_limit: wall-clock budget (seconds)

        Returns:
            [num_macros, 2] optimized positions
        """
        start = time.time()

        fixed_pos = init_positions.clone().to(self.device)
        positions = init_positions.clone().detach().to(self.device)

        movable_hard = self.movable_mask[:self.num_hard]
        if not movable_hard.any():
            return positions

        positions.requires_grad_(True)

        # Learning rate proportional to canvas size
        canvas_dim = min(self.canvas_w, self.canvas_h)
        canvas_area = self.canvas_w * self.canvas_h
        lr = canvas_dim * 0.005
        optimizer = torch.optim.Adam([positions], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iters, eta_min=lr * 0.02
        )

        phase1_iters = int(num_iters * 0.8)
        best_positions = positions.detach().clone()
        best_score = float('inf')
        final_it = 0

        for it in range(num_iters):
            if time.time() - start > time_limit:
                break
            final_it = it + 1

            optimizer.zero_grad()

            if it < phase1_iters:
                # Phase 1: Use proven total_loss (same as GNN training)
                loss, ld = total_loss(
                    positions, self.graph, self.benchmark,
                    epoch=it, max_epochs=phase1_iters,
                )
            else:
                # Phase 2: Fine-grid density + strong overlap
                t2 = (it - phase1_iters) / max(num_iters - phase1_iters - 1, 1)
                gamma = 10.0 + t2 * 10.0

                wl = wirelength_loss(positions, self.graph, self.benchmark, gamma=gamma)
                den = density_loss(positions, self.benchmark, grid_size=24)
                ov = overlap_loss(positions, self.benchmark)

                w_ov = 50.0 + t2 * 150.0  # 50 -> 200
                loss = wl / canvas_area + 5.0 * den + w_ov * ov / canvas_area
                ld = {'wirelength': (wl / canvas_area).item(),
                      'density': den.item(),
                      'overlap': (ov / canvas_area).item()}

            loss.backward()

            # Zero gradients for fixed macros
            if positions.grad is not None:
                positions.grad.data[self.fixed_mask] = 0

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([positions], max_norm=canvas_dim)

            optimizer.step()
            scheduler.step()

            # Hard clamp
            with torch.no_grad():
                self._clamp(positions, fixed_pos)

            # Track best positions
            with torch.no_grad():
                ov_val = ld.get('overlap', 0.0)
                wl_val = ld.get('wirelength', 0.0)
                den_val = ld.get('density', 0.0)
                score = wl_val + 3.0 * den_val + 10.0 * ov_val
                if score < best_score:
                    best_score = score
                    best_positions = positions.detach().clone()

            # Logging
            if it == 0 or (it + 1) % 100 == 0 or it == num_iters - 1:
                phase = "P1" if it < phase1_iters else "P2"
                elapsed = time.time() - start
                print(f"      ePlace iter {it+1}/{num_iters} [{phase}]: "
                      f"wl={ld.get('wirelength', 0):.6f} "
                      f"den={ld.get('density', 0):.4f} "
                      f"ov={ld.get('overlap', 0):.6f} "
                      f"[{elapsed:.1f}s]", flush=True)

        elapsed = time.time() - start
        print(f"    ePlace done: {final_it} iters [{elapsed:.1f}s]")

        return best_positions
