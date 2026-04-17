"""
Legalization and post-legalization refinement for macro placement.

Converts continuous GNN output into a valid placement with zero overlaps,
then refines via coordinate descent, density equalization, and simulated annealing.
"""

import torch
import time
import random
import math
import numpy as np


def _overlaps_any(px, py, pw, ph, placed_pos, placed_sizes):
    """Check if macro at (px,py) with size (pw,ph) overlaps any placed macro.
    Uses a small positive gap to prevent float-precision false overlaps."""
    GAP = 0.01
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


def _overlaps_any_np(px, py, pw, ph, pp_np, ps_np):
    """Vectorized overlap check using numpy arrays. ~50x faster than Python loop."""
    if len(pp_np) == 0:
        return False
    dx = np.abs(px - pp_np[:, 0])
    dy = np.abs(py - pp_np[:, 1])
    min_sep_x = (pw + ps_np[:, 0]) / 2.0 + 0.01
    min_sep_y = (ph + ps_np[:, 1]) / 2.0 + 0.01
    return bool(np.any((dx < min_sep_x) & (dy < min_sep_y)))


def _find_nearest_valid(px, py, w, h, placed_pos, placed_sizes, canvas_w, canvas_h,
                        pp_np=None, ps_np=None):
    """
    Find the nearest non-overlapping, in-bounds position to (px, py).
    Uses numpy-vectorized overlap checks for speed.

    If pp_np/ps_np are provided, uses them directly (avoids repeated conversion).
    Otherwise converts placed_pos/placed_sizes to numpy.
    """
    hw, hh = w / 2, h / 2

    px = max(hw, min(canvas_w - hw, px))
    py = max(hh, min(canvas_h - hh, py))

    # Use pre-built numpy arrays if provided, otherwise convert
    if pp_np is None:
        if len(placed_pos) > 0:
            pp_np = np.array(placed_pos)
            ps_np = np.array(placed_sizes)
        else:
            return (px, py)
    elif len(pp_np) == 0:
        return (px, py)

    # Try original position
    if not _overlaps_any_np(px, py, w, h, pp_np, ps_np):
        return (px, py)

    best_pos = None
    best_dist = float('inf')

    # Strategy 1: Push away from overlapping macros (vectorized detection)
    dx_all = np.abs(px - pp_np[:, 0])
    dy_all = np.abs(py - pp_np[:, 1])
    sep_x = (w + ps_np[:, 0]) / 2.0
    sep_y = (h + ps_np[:, 1]) / 2.0
    overlap_mask = (dx_all < sep_x) & (dy_all < sep_y)

    for i in np.where(overlap_mask)[0]:
        qx, qy = pp_np[i]
        qw, qh = ps_np[i]
        gap = 0.02
        shifts = [
            (qx + (qw + w) / 2.0 + gap, py),
            (qx - (qw + w) / 2.0 - gap, py),
            (px, qy + (qh + h) / 2.0 + gap),
            (px, qy - (qh + h) / 2.0 - gap),
        ]
        for nx, ny in shifts:
            nx = max(hw, min(canvas_w - hw, nx))
            ny = max(hh, min(canvas_h - hh, ny))
            if not _overlaps_any_np(nx, ny, w, h, pp_np, ps_np):
                dist = (nx - px) ** 2 + (ny - py) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (nx, ny)

    if best_pos is not None:
        return best_pos

    # Strategy 2: Vectorized spiral search — check entire ring at once
    step = min(w, h) * 0.3
    for ring in range(1, 200):
        radius = ring * step
        num_points = max(8, int(2 * math.pi * radius / step))
        angles = np.arange(num_points) * (2 * math.pi / num_points) + ring * 0.1
        xs = np.clip(px + radius * np.cos(angles), hw, canvas_w - hw)
        ys = np.clip(py + radius * np.sin(angles), hh, canvas_h - hh)

        # Batch overlap check: [num_points, num_placed]
        dxs = np.abs(xs[:, np.newaxis] - pp_np[:, 0])
        dys = np.abs(ys[:, np.newaxis] - pp_np[:, 1])
        min_sep_x = (w + ps_np[:, 0]) / 2.0 + 0.01
        min_sep_y = (h + ps_np[:, 1]) / 2.0 + 0.01
        has_overlap = np.any((dxs < min_sep_x) & (dys < min_sep_y), axis=1)
        valid = ~has_overlap

        if np.any(valid):
            # Pick the first valid point (closest along spiral)
            idx = np.argmax(valid)
            return (float(xs[idx]), float(ys[idx]))

    # Strategy 3: Exhaustive grid search as final fallback
    step_x = w * 1.01
    step_y = h * 1.01
    for gy_i in range(int(canvas_h / step_y) + 1):
        for gx_i in range(int(canvas_w / step_x) + 1):
            nx = gx_i * step_x + hw
            ny = gy_i * step_y + hh
            if nx + hw <= canvas_w + 0.01 and ny + hh <= canvas_h + 0.01:
                if not _overlaps_any_np(nx, ny, w, h, pp_np, ps_np):
                    return (nx, ny)

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


def density_equalize(
    positions: torch.Tensor,
    benchmark,
    compute_cost_fn,
    plc,
    time_limit: float = 300.0,
) -> torch.Tensor:
    """
    Density equalization: move macros from high-density regions to low-density ones.

    Computes grid cell density, identifies macros in the densest cells,
    and tries relocating them to less-dense cells. Each move is validated
    with the full proxy cost to ensure improvement.
    """
    best = positions.clone().detach()
    num_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu()
    fixed = benchmark.macro_fixed.cpu()
    grid_col = benchmark.grid_cols
    grid_row = benchmark.grid_rows

    # Use fast eval (skip congestion) for DenEq — focuses on density improvement
    best_cost = compute_cost_fn(best, benchmark, plc, skip_congestion=True)['proxy_cost']
    start_time = time.time()
    improvements = 0
    evals = 0

    cell_w = canvas_w / grid_col
    cell_h = canvas_h / grid_row

    for round_i in range(5):
        if time.time() - start_time > time_limit:
            break

        # Compute density grid (hard macros only for targeting)
        pos_np = best[:num_hard].numpy()
        sizes_np = sizes[:num_hard].numpy()
        density = np.zeros((grid_row, grid_col))

        for i in range(num_hard):
            x, y = pos_np[i]
            w, h = sizes_np[i]
            x0 = max(0, int((x - w / 2) / cell_w))
            x1 = min(grid_col - 1, int((x + w / 2) / cell_w))
            y0 = max(0, int((y - h / 2) / cell_h))
            y1 = min(grid_row - 1, int((y + h / 2) / cell_h))
            for r in range(y0, y1 + 1):
                for c in range(x0, x1 + 1):
                    # Overlap area between macro and cell
                    ox = max(0, min(x + w / 2, (c + 1) * cell_w) - max(x - w / 2, c * cell_w))
                    oy = max(0, min(y + h / 2, (r + 1) * cell_h) - max(y - h / 2, r * cell_h))
                    density[r, c] += ox * oy / (cell_w * cell_h)

        # Find top 10% densest cells
        flat_den = density.flatten()
        k = max(1, len(flat_den) // 10)
        top_indices = np.argsort(flat_den)[-k:]
        avg_density = flat_den.mean()

        # Find macros in the densest cells
        dense_macros = set()
        for flat_idx in top_indices:
            r, c = divmod(flat_idx, grid_col)
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            for i in range(num_hard):
                if fixed[i]:
                    continue
                x, y = pos_np[i]
                w_i, h_i = sizes_np[i]
                if abs(x - cx) < (w_i / 2 + cell_w) and abs(y - cy) < (h_i / 2 + cell_h):
                    dense_macros.add(i)

        # Find low-density target regions (below average)
        low_cells = [(r, c) for r in range(grid_row) for c in range(grid_col)
                     if density[r, c] < avg_density * 0.5]
        if not low_cells:
            low_cells = [(r, c) for r in range(grid_row) for c in range(grid_col)
                         if density[r, c] < avg_density]

        # Try moving each dense macro toward a low-density cell
        dense_list = list(dense_macros)
        random.shuffle(dense_list)

        placed_pos = []
        placed_sizes = []
        for j in range(num_hard):
            placed_pos.append((best[j, 0].item(), best[j, 1].item()))
            placed_sizes.append((sizes[j, 0].item(), sizes[j, 1].item()))

        for idx in dense_list:
            if time.time() - start_time > time_limit:
                break

            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()
            cur_x = best[idx, 0].item()
            cur_y = best[idx, 1].item()

            # Build placed list excluding this macro (as numpy arrays)
            placed_ex = [(placed_pos[j], placed_sizes[j]) for j in range(num_hard) if j != idx]
            pp = [p[0] for p in placed_ex]
            ps = [p[1] for p in placed_ex]
            pp_np_de = np.array(pp) if pp else np.empty((0, 2))
            ps_np_de = np.array(ps) if ps else np.empty((0, 2))

            # Sort low-density cells: prefer cells near current position
            # (less displacement = less wirelength impact)
            scored_cells = [(r, c, abs((c+0.5)*cell_w - cur_x) + abs((r+0.5)*cell_h - cur_y))
                           for r, c in low_cells]
            scored_cells.sort(key=lambda x: x[2])
            targets = [(r, c) for r, c, _ in scored_cells[:12]]

            best_trial = None
            best_trial_cost = best_cost

            for (tr, tc) in targets:
                if time.time() - start_time > time_limit:
                    break
                target_x = (tc + 0.5) * cell_w
                target_y = (tr + 0.5) * cell_h
                target_x = max(w / 2, min(canvas_w - w / 2, target_x))
                target_y = max(h / 2, min(canvas_h - h / 2, target_y))

                new_pos = _find_nearest_valid(
                    target_x, target_y, w, h, pp, ps, canvas_w, canvas_h,
                    pp_np=pp_np_de, ps_np=ps_np_de,
                )
                if _overlaps_any_np(new_pos[0], new_pos[1], w, h, pp_np_de, ps_np_de):
                    continue
                if abs(new_pos[0] - cur_x) < 0.1 and abs(new_pos[1] - cur_y) < 0.1:
                    continue

                trial = best.clone()
                trial[idx, 0] = new_pos[0]
                trial[idx, 1] = new_pos[1]
                trial_cost = compute_cost_fn(trial, benchmark, plc, skip_congestion=True)['proxy_cost']
                evals += 1

                if trial_cost < best_trial_cost:
                    best_trial_cost = trial_cost
                    best_trial = trial
                    break  # Accept first improvement

            if best_trial is not None:
                best = best_trial
                best_cost = best_trial_cost
                improvements += 1
                # Update placed_pos
                placed_pos[idx] = (best[idx, 0].item(), best[idx, 1].item())

        elapsed = time.time() - start_time
        print(f"    DenEq round {round_i+1}: cost={best_cost:.4f} "
              f"improvements={improvements} evals={evals} [{elapsed:.1f}s]", flush=True)

    return best


def _hpwl_delta(idx, old_x, old_y, new_x, new_y, positions, macro_nets, benchmark):
    """
    Compute change in HPWL if macro idx moves from (old_x,old_y) to (new_x,new_y).
    Negative means improvement. Only considers nets connected to this macro.
    """
    num_macros = benchmark.num_macros
    port_pos = benchmark.port_positions
    delta = 0.0
    for net_idx in macro_nets[idx]:
        nodes = benchmark.net_nodes[net_idx].tolist()
        old_min_x = old_max_x = old_x
        old_min_y = old_max_y = old_y
        new_min_x = new_max_x = new_x
        new_min_y = new_max_y = new_y
        for n in nodes:
            if n == idx:
                continue
            if n < num_macros:
                px = positions[n, 0].item()
                py = positions[n, 1].item()
            elif n < num_macros + port_pos.size(0):
                px = port_pos[n - num_macros, 0].item()
                py = port_pos[n - num_macros, 1].item()
            else:
                continue
            old_min_x = min(old_min_x, px); old_max_x = max(old_max_x, px)
            old_min_y = min(old_min_y, py); old_max_y = max(old_max_y, py)
            new_min_x = min(new_min_x, px); new_max_x = max(new_max_x, px)
            new_min_y = min(new_min_y, py); new_max_y = max(new_max_y, py)
        old_hpwl = (old_max_x - old_min_x) + (old_max_y - old_min_y)
        new_hpwl = (new_max_x - new_min_x) + (new_max_y - new_min_y)
        delta += new_hpwl - old_hpwl
    return delta


def coordinate_descent_refine(
    positions: torch.Tensor,
    benchmark,
    compute_cost_fn,
    plc,
    num_sweeps: int = 20,
    time_limit: float = 2700.0,
    skip_congestion: bool = True,
) -> torch.Tensor:
    """
    Coordinate descent: for each macro, try multiple candidate positions
    (median of connected pins, random perturbations, weighted centroid).
    Uses cheap HPWL pre-filter + configurable congestion skipping for
    fast proxy cost evaluations.
    """
    best = positions.clone().detach()
    num_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu()
    fixed = benchmark.macro_fixed.cpu()

    best_cost = compute_cost_fn(best, benchmark, plc, skip_congestion=skip_congestion)['proxy_cost']

    # Build net-to-macro adjacency
    macro_nets = [[] for _ in range(benchmark.num_macros)]
    for net_idx in range(benchmark.num_nets):
        for node_idx in benchmark.net_nodes[net_idx].tolist():
            if node_idx < benchmark.num_macros:
                macro_nets[node_idx].append(net_idx)

    start_time = time.time()
    total_improvements = 0
    proxy_evals = 0
    no_improve_count = 0

    grid_col = getattr(benchmark, 'grid_cols', 32)
    grid_row = getattr(benchmark, 'grid_rows', 32)
    cell_w_cd = canvas_w / grid_col
    cell_h_cd = canvas_h / grid_row

    for sweep in range(num_sweeps):
        if time.time() - start_time > time_limit:
            break

        # Compute density grid for density-aware candidates
        pos_np_cd = best[:num_hard].detach().numpy()
        sizes_np_cd = sizes[:num_hard].numpy()
        den_grid = np.zeros((grid_row, grid_col))
        for mi in range(num_hard):
            mx, my = pos_np_cd[mi]
            mw, mh = sizes_np_cd[mi]
            c0 = max(0, int((mx - mw/2) / cell_w_cd))
            c1 = min(grid_col-1, int((mx + mw/2) / cell_w_cd))
            r0 = max(0, int((my - mh/2) / cell_h_cd))
            r1 = min(grid_row-1, int((my + mh/2) / cell_h_cd))
            for rr in range(r0, r1+1):
                for cc in range(c0, c1+1):
                    ox = max(0, min(mx+mw/2, (cc+1)*cell_w_cd) - max(mx-mw/2, cc*cell_w_cd))
                    oy = max(0, min(my+mh/2, (rr+1)*cell_h_cd) - max(my-mh/2, rr*cell_h_cd))
                    den_grid[rr, cc] += ox * oy / (cell_w_cd * cell_h_cd)
        den_avg = den_grid.mean()

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

            # Build placed list excluding current macro (as numpy arrays)
            placed_pos = []
            placed_sizes = []
            for j in range(num_hard):
                if j != idx:
                    placed_pos.append((best[j, 0].item(), best[j, 1].item()))
                    placed_sizes.append((sizes[j, 0].item(), sizes[j, 1].item()))
            pp_np_cd = np.array(placed_pos) if placed_pos else np.empty((0, 2))
            ps_np_cd = np.array(placed_sizes) if placed_sizes else np.empty((0, 2))

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
                for alpha in [0.25, 0.5, 0.75]:
                    candidates.append((
                        cur_x + alpha * (med_x - cur_x),
                        cur_y + alpha * (med_y - cur_y),
                    ))

            # Density-aware: move toward low-density region nearby
            cur_r = min(grid_row-1, max(0, int(cur_y / cell_h_cd)))
            cur_c = min(grid_col-1, max(0, int(cur_x / cell_w_cd)))
            if den_grid[cur_r, cur_c] > den_avg:
                # Find nearest low-density cell
                low_r, low_c = cur_r, cur_c
                best_score_den = float('inf')
                search_r = max(3, int(h / cell_h_cd) + 2)
                search_c = max(3, int(w / cell_w_cd) + 2)
                for dr in range(-search_r, search_r+1):
                    for dc in range(-search_c, search_c+1):
                        nr, nc = cur_r+dr, cur_c+dc
                        if 0 <= nr < grid_row and 0 <= nc < grid_col:
                            # Score: density + distance penalty
                            dist = (abs(dr) + abs(dc)) * 0.1
                            score_den = den_grid[nr, nc] + dist
                            if score_den < best_score_den:
                                best_score_den = score_den
                                low_r, low_c = nr, nc
                candidates.append(((low_c+0.5)*cell_w_cd, (low_r+0.5)*cell_h_cd))

            # Random perturbations (can improve density/congestion)
            step = min(w, h) * 0.5
            for _ in range(3):
                candidates.append((
                    cur_x + random.gauss(0, step),
                    cur_y + random.gauss(0, step),
                ))

            # Collect valid candidates with HPWL pre-filter
            # Fast path: check candidate directly, skip expensive spiral search
            valid_candidates = []
            for target_x, target_y in candidates:
                target_x = max(w / 2, min(canvas_w - w / 2, target_x))
                target_y = max(h / 2, min(canvas_h - h / 2, target_y))

                # Check if candidate position is directly valid (no overlap)
                if not _overlaps_any_np(target_x, target_y, w, h, pp_np_cd, ps_np_cd):
                    new_pos = (target_x, target_y)
                else:
                    # Try push-away from nearest overlapping macro (cheap)
                    new_pos = _find_nearest_valid(
                        target_x, target_y, w, h,
                        placed_pos, placed_sizes, canvas_w, canvas_h,
                        pp_np=pp_np_cd, ps_np=ps_np_cd,
                    )
                    if _overlaps_any_np(new_pos[0], new_pos[1], w, h, pp_np_cd, ps_np_cd):
                        continue

                # Skip if barely moved
                if abs(new_pos[0] - cur_x) < 0.1 and abs(new_pos[1] - cur_y) < 0.1:
                    continue

                # Cheap HPWL delta check
                hpwl_d = _hpwl_delta(
                    idx, cur_x, cur_y, new_pos[0], new_pos[1],
                    best, macro_nets, benchmark,
                )
                valid_candidates.append((new_pos, hpwl_d))

            if not valid_candidates:
                continue

            # Sort by HPWL improvement, only evaluate the best ones with
            # expensive proxy cost
            valid_candidates.sort(key=lambda x: x[1])

            # Evaluate top candidates (those that improve or barely hurt HPWL)
            best_trial = None
            best_trial_cost = best_cost
            max_evals = 3  # Limit expensive evaluations per macro

            evals_done = 0
            for (new_pos, hpwl_d) in valid_candidates:
                if evals_done >= max_evals:
                    break
                # Skip candidates that significantly worsen HPWL
                # (allow small HPWL increase since density/congestion may improve)
                if hpwl_d > 0 and evals_done > 0:
                    break

                trial = best.clone()
                trial[idx, 0] = new_pos[0]
                trial[idx, 1] = new_pos[1]

                trial_cost = compute_cost_fn(trial, benchmark, plc, skip_congestion=skip_congestion)['proxy_cost']
                proxy_evals += 1
                evals_done += 1
                if trial_cost < best_trial_cost:
                    best_trial_cost = trial_cost
                    best_trial = trial

            if best_trial is not None:
                best = best_trial
                best_cost = best_trial_cost
                improved = True
                total_improvements += 1

        elapsed = time.time() - start_time
        label = "CD" if skip_congestion else "CD-full"
        print(f"    {label} sweep {sweep+1}: cost={best_cost:.4f} "
              f"improvements={total_improvements} evals={proxy_evals} "
              f"[{elapsed:.1f}s]", flush=True)

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
