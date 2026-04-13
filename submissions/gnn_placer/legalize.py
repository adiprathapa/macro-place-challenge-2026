"""
Legalization and post-legalization refinement for macro placement.

Converts continuous GNN output into a valid placement with zero overlaps,
then refines via coordinate descent and simulated annealing.
"""

import torch
import time
import random
import math


def _overlaps_any(px, py, pw, ph, placed_pos, placed_sizes):
    """Check if macro at (px,py) with size (pw,ph) overlaps any placed macro.
    Uses a small positive gap to prevent float-precision false overlaps."""
    GAP = 0.01  # Small gap to prevent touching-edge overlaps
    for i in range(len(placed_pos)):
        qx, qy = placed_pos[i]
        qw, qh = placed_sizes[i]
        dx = abs(px - qx)
        dy = abs(py - qy)
        min_sep_x = (pw + qw) / 2.0 + GAP
        min_sep_y = (ph + qh) / 2.0 + GAP
        if dx < min_sep_x and dy < min_sep_y:
            return True
    return False


def _find_nearest_valid(px, py, w, h, placed_pos, placed_sizes, canvas_w, canvas_h):
    """
    Find the nearest non-overlapping, in-bounds position to (px, py).

    Uses a multi-strategy search: spiral, then cardinal pushes, then full grid.
    """
    hw, hh = w / 2, h / 2

    # Clamp to canvas first
    px = max(hw, min(canvas_w - hw, px))
    py = max(hh, min(canvas_h - hh, py))

    # Try original position
    if not _overlaps_any(px, py, w, h, placed_pos, placed_sizes):
        return (px, py)

    best_pos = None
    best_dist = float('inf')

    # Strategy 1: Try shifting away from each overlapping macro
    for i in range(len(placed_pos)):
        qx, qy = placed_pos[i]
        qw, qh = placed_sizes[i]
        dx_val = abs(px - qx)
        dy_val = abs(py - qy)
        if dx_val < (w + qw) / 2.0 and dy_val < (h + qh) / 2.0:
            # This macro overlaps; try pushing in each cardinal direction
            gap = 0.02
            shifts = [
                (qx + (qw + w) / 2.0 + gap, py),   # push right
                (qx - (qw + w) / 2.0 - gap, py),   # push left
                (px, qy + (qh + h) / 2.0 + gap),   # push up
                (px, qy - (qh + h) / 2.0 - gap),   # push down
            ]
            for nx, ny in shifts:
                nx = max(hw, min(canvas_w - hw, nx))
                ny = max(hh, min(canvas_h - hh, ny))
                if not _overlaps_any(nx, ny, w, h, placed_pos, placed_sizes):
                    dist = (nx - px) ** 2 + (ny - py) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nx, ny)

    if best_pos is not None:
        return best_pos

    # Strategy 2: Spiral search with finer resolution
    step = min(w, h) * 0.3
    for ring in range(1, 200):
        radius = ring * step
        num_points = max(8, int(2 * math.pi * radius / step))
        for k in range(num_points):
            angle = 2 * math.pi * k / num_points + ring * 0.1
            nx = px + radius * math.cos(angle)
            ny = py + radius * math.sin(angle)
            nx = max(hw, min(canvas_w - hw, nx))
            ny = max(hh, min(canvas_h - hh, ny))
            if not _overlaps_any(nx, ny, w, h, placed_pos, placed_sizes):
                return (nx, ny)

    # Strategy 3: Exhaustive grid search as final fallback
    step_x = w * 1.01
    step_y = h * 1.01
    for gy_i in range(int(canvas_h / step_y) + 1):
        for gx_i in range(int(canvas_w / step_x) + 1):
            nx = gx_i * step_x + hw
            ny = gy_i * step_y + hh
            if nx + hw <= canvas_w + 0.01 and ny + hh <= canvas_h + 0.01:
                if not _overlaps_any(nx, ny, w, h, placed_pos, placed_sizes):
                    return (nx, ny)

    # Absolute last resort
    return (max(hw, min(canvas_w - hw, px)),
            max(hh, min(canvas_h - hh, py)))


def legalize(
    positions: torch.Tensor,
    benchmark,
) -> torch.Tensor:
    """
    Greedy displacement legalization.

    Sort hard macros by area (largest first), place greedily with
    search for nearest non-overlapping positions. Guarantees zero
    overlaps if canvas has sufficient area.
    """
    legal = positions.clone().detach()
    num_hard = benchmark.num_hard_macros
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    sizes = benchmark.macro_sizes.cpu()
    fixed = benchmark.macro_fixed.cpu()

    # Collect fixed macros as pre-placed
    placed_pos = []
    placed_sizes = []

    for i in range(num_hard):
        if fixed[i]:
            placed_pos.append((legal[i, 0].item(), legal[i, 1].item()))
            placed_sizes.append((sizes[i, 0].item(), sizes[i, 1].item()))

    # Sort movable hard macros by area (largest first)
    movable_hard = [(i, sizes[i, 0].item() * sizes[i, 1].item())
                    for i in range(num_hard) if not fixed[i]]
    movable_hard.sort(key=lambda x: -x[1])

    for idx, _ in movable_hard:
        w = sizes[idx, 0].item()
        h = sizes[idx, 1].item()
        target_x = legal[idx, 0].item()
        target_y = legal[idx, 1].item()

        pos = _find_nearest_valid(
            target_x, target_y, w, h,
            placed_pos, placed_sizes, canvas_w, canvas_h,
        )
        legal[idx, 0] = pos[0]
        legal[idx, 1] = pos[1]
        placed_pos.append(pos)
        placed_sizes.append((w, h))

    return legal


def coordinate_descent_refine(
    positions: torch.Tensor,
    benchmark,
    compute_cost_fn,
    plc,
    num_sweeps: int = 20,
    time_limit: float = 2700.0,
) -> torch.Tensor:
    """
    Coordinate descent: for each macro, try multiple candidate positions
    (median of connected pins, random perturbations, weighted centroid)
    and keep the best improvement.
    """
    best = positions.clone().detach()
    num_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu()
    fixed = benchmark.macro_fixed.cpu()

    best_cost = compute_cost_fn(best, benchmark, plc)['proxy_cost']

    # Build net-to-macro adjacency
    macro_nets = [[] for _ in range(benchmark.num_macros)]
    for net_idx in range(benchmark.num_nets):
        for node_idx in benchmark.net_nodes[net_idx].tolist():
            if node_idx < benchmark.num_macros:
                macro_nets[node_idx].append(net_idx)

    start_time = time.time()
    total_improvements = 0
    no_improve_count = 0

    for sweep in range(num_sweeps):
        if time.time() - start_time > time_limit:
            break

        improved = False
        movable_indices = [i for i in range(num_hard) if not fixed[i]]
        random.shuffle(movable_indices)

        for idx in movable_indices:
            if time.time() - start_time > time_limit:
                break

            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()
            cur_x = best[idx, 0].item()
            cur_y = best[idx, 1].item()

            # Compute connected pin positions
            connected_x = []
            connected_y = []
            for net_idx in macro_nets[idx]:
                for node_idx in benchmark.net_nodes[net_idx].tolist():
                    if node_idx == idx:
                        continue
                    if node_idx < benchmark.num_macros:
                        connected_x.append(best[node_idx, 0].item())
                        connected_y.append(best[node_idx, 1].item())
                    elif node_idx < benchmark.num_macros + benchmark.port_positions.size(0):
                        port_idx = node_idx - benchmark.num_macros
                        connected_x.append(benchmark.port_positions[port_idx, 0].item())
                        connected_y.append(benchmark.port_positions[port_idx, 1].item())

            # Build placed list excluding current macro
            placed_pos = []
            placed_sizes = []
            for j in range(num_hard):
                if j != idx:
                    placed_pos.append((best[j, 0].item(), best[j, 1].item()))
                    placed_sizes.append((sizes[j, 0].item(), sizes[j, 1].item()))

            # Generate candidate target positions
            candidates = []

            if connected_x:
                connected_x.sort()
                connected_y.sort()
                med_x = connected_x[len(connected_x) // 2]
                med_y = connected_y[len(connected_y) // 2]
                candidates.append((med_x, med_y))

                # Weighted centroid (mean)
                mean_x = sum(connected_x) / len(connected_x)
                mean_y = sum(connected_y) / len(connected_y)
                candidates.append((mean_x, mean_y))

                # Interpolate between current and median
                for alpha in [0.3, 0.5, 0.7]:
                    candidates.append((
                        cur_x + alpha * (med_x - cur_x),
                        cur_y + alpha * (med_y - cur_y),
                    ))

            # Try each candidate
            best_trial = None
            best_trial_cost = best_cost

            for target_x, target_y in candidates:
                target_x = max(w / 2, min(canvas_w - w / 2, target_x))
                target_y = max(h / 2, min(canvas_h - h / 2, target_y))

                new_pos = _find_nearest_valid(
                    target_x, target_y, w, h,
                    placed_pos, placed_sizes, canvas_w, canvas_h,
                )

                if _overlaps_any(new_pos[0], new_pos[1], w, h, placed_pos, placed_sizes):
                    continue

                # Skip if barely moved
                if abs(new_pos[0] - cur_x) < 0.1 and abs(new_pos[1] - cur_y) < 0.1:
                    continue

                trial = best.clone()
                trial[idx, 0] = new_pos[0]
                trial[idx, 1] = new_pos[1]

                trial_cost = compute_cost_fn(trial, benchmark, plc)['proxy_cost']
                if trial_cost < best_trial_cost:
                    best_trial_cost = trial_cost
                    best_trial = trial

            if best_trial is not None:
                best = best_trial
                best_cost = best_trial_cost
                improved = True
                total_improvements += 1

        elapsed = time.time() - start_time
        print(f"    CD sweep {sweep+1}: cost={best_cost:.4f} "
              f"improvements={total_improvements} [{elapsed:.1f}s]", flush=True)

        if not improved:
            no_improve_count += 1
            if no_improve_count >= 2:
                print(f"    CD converged (no improvement for 2 sweeps)")
                break
        else:
            no_improve_count = 0

    return best


def sa_refine(
    positions: torch.Tensor,
    benchmark,
    compute_cost_fn,
    plc,
    time_limit: float = 600.0,
) -> torch.Tensor:
    """
    Fast simulated annealing using lightweight numpy-based HPWL cost.

    Uses three move types: shift, swap, and move-toward-neighbor.
    Overlap checking is vectorized O(N) per move.
    """
    import numpy as np

    num_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float64)
    fixed_np = benchmark.macro_fixed[:num_hard].cpu().numpy()

    movable_idx = np.where(~fixed_np)[0]
    if len(movable_idx) == 0:
        return positions

    pos = positions[:num_hard].clone().detach().cpu().numpy().astype(np.float64)
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2

    # Precompute pairwise separation thresholds
    sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

    # Build net connectivity for HPWL cost (hard macros only)
    edge_dict = {}
    for net_idx in range(benchmark.num_nets):
        nodes = benchmark.net_nodes[net_idx].tolist()
        hard_nodes = [n for n in nodes if n < num_hard]
        if len(hard_nodes) >= 2:
            w = 1.0 / (len(hard_nodes) - 1)
            for i_idx in range(len(hard_nodes)):
                for j_idx in range(i_idx + 1, len(hard_nodes)):
                    pair = (hard_nodes[i_idx], hard_nodes[j_idx])
                    if pair[0] > pair[1]:
                        pair = (pair[1], pair[0])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w

    if not edge_dict:
        return positions

    edges = np.array(list(edge_dict.keys()), dtype=np.int64)
    edge_weights = np.array([edge_dict[tuple(e)] for e in edges], dtype=np.float64)

    # Build neighbor lists for move-toward-neighbor
    neighbors = [[] for _ in range(num_hard)]
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    def wl_cost():
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        return (edge_weights * (dx + dy)).sum()

    def check_overlap(idx):
        gap = 0.05
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        overlaps = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
        overlaps[idx] = False
        return overlaps.any()

    current_cost = wl_cost()
    best_pos = pos.copy()
    best_cost = current_cost

    T_start = max(canvas_w, canvas_h) * 0.15
    T_end = max(canvas_w, canvas_h) * 0.001

    start_time = time.time()
    max_steps = 500000
    step = 0
    accepted = 0

    while step < max_steps and time.time() - start_time < time_limit:
        step += 1
        frac = step / max_steps
        T = T_start * (T_end / T_start) ** frac

        move_type = random.random()
        i = random.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        if move_type < 0.5:
            # SHIFT: random perturbation
            shift = T * (0.3 + 0.7 * (1 - frac))
            pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift), half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift), half_h[i], canvas_h - half_h[i])

        elif move_type < 0.8:
            # SWAP: swap positions with another macro
            if neighbors[i] and random.random() < 0.7:
                cands = [j for j in neighbors[i] if not fixed_np[j]]
                j = random.choice(cands) if cands else random.choice(movable_idx)
            else:
                j = random.choice(movable_idx)
            if i != j:
                old_jx, old_jy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(old_jx, half_w[i], canvas_w - half_w[i])
                pos[i, 1] = np.clip(old_jy, half_h[i], canvas_h - half_h[i])
                pos[j, 0] = np.clip(old_x, half_w[j], canvas_w - half_w[j])
                pos[j, 1] = np.clip(old_y, half_h[j], canvas_h - half_h[j])
                if check_overlap(i) or check_overlap(j):
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy
                    continue
                new_cost = wl_cost()
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    accepted += 1
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_pos = pos.copy()
                else:
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy
                continue

        else:
            # MOVE TOWARD NEIGHBOR
            if neighbors[i]:
                j = random.choice(neighbors[i])
                alpha = random.uniform(0.05, 0.3)
                pos[i, 0] = np.clip(pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]),
                                     half_w[i], canvas_w - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]),
                                     half_h[i], canvas_h - half_h[i])
            else:
                continue

        # Check overlap for single-macro moves
        if check_overlap(i):
            pos[i, 0] = old_x; pos[i, 1] = old_y
            continue

        new_cost = wl_cost()
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            accepted += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = pos.copy()
        else:
            pos[i, 0] = old_x; pos[i, 1] = old_y

        if step % 50000 == 0:
            elapsed = time.time() - start_time
            rate = accepted / step if step > 0 else 0
            print(f"    SA step {step}: wl={current_cost:.1f} best_wl={best_cost:.1f} "
                  f"T={T:.2f} accept={rate:.3f} [{elapsed:.1f}s]", flush=True)

    elapsed = time.time() - start_time
    rate = accepted / step if step > 0 else 0
    print(f"    SA done: {step} steps, accept={rate:.3f}, "
          f"best_wl={best_cost:.1f} [{elapsed:.1f}s]", flush=True)

    # Write back to full positions tensor
    result = positions.clone().detach()
    result[:num_hard] = torch.tensor(best_pos, dtype=torch.float32)
    return result
