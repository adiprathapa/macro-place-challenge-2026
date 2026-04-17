"""
Differentiable loss functions for GNN-based macro placement.

All losses are smooth approximations of the true proxy cost components,
designed for gradient-based optimization. Fully vectorized for speed.
"""

import torch
import torch.nn.functional as F


def wirelength_loss(
    positions: torch.Tensor,
    graph,
    benchmark,
    gamma: float = 10.0,
) -> torch.Tensor:
    """
    Differentiable HPWL approximation using log-sum-exp (fully vectorized).
    """
    num_ports = benchmark.port_positions.size(0)
    device = positions.device

    if num_ports > 0:
        port_pos = benchmark.port_positions.to(device)
        all_positions = torch.cat([positions, port_pos], dim=0)
    else:
        all_positions = positions

    indices = graph.pin_node_indices.clamp(0, all_positions.size(0) - 1)
    pin_positions = all_positions[indices]  # [num_nets, max_degree, 2]
    mask = graph.net_mask  # [num_nets, max_degree]

    large_val = 1e6

    # Process both dimensions at once
    # pin_positions: [num_nets, max_degree, 2]
    coords = pin_positions  # [N, D, 2]
    mask_2d = mask.unsqueeze(-1).expand_as(coords)  # [N, D, 2]

    coords_for_max = coords.masked_fill(~mask_2d, -large_val)
    coords_for_min = coords.masked_fill(~mask_2d, large_val)

    # [num_nets, 2]
    lse_max = torch.logsumexp(gamma * coords_for_max, dim=1) / gamma
    lse_min = -torch.logsumexp(-gamma * coords_for_min, dim=1) / gamma

    hpwl = (lse_max - lse_min).sum(dim=1)  # [num_nets]

    return (graph.net_weights * hpwl).sum()


def density_loss(
    positions: torch.Tensor,
    benchmark,
    grid_size: int = 12,
) -> torch.Tensor:
    """
    Fully vectorized grid-based density loss (hard macros only).
    Uses L2 excess density penalty.
    """
    device = positions.device
    num_hard = benchmark.num_hard_macros
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height

    grid_col = grid_size
    grid_row = grid_size

    hard_pos = positions[:num_hard]
    hard_sizes = benchmark.macro_sizes[:num_hard].to(device)

    cell_w = canvas_w / grid_col
    cell_h = canvas_h / grid_row

    bin_x_lo = torch.arange(grid_col, device=device).float() * cell_w
    bin_x_hi = bin_x_lo + cell_w
    bin_y_lo = torch.arange(grid_row, device=device).float() * cell_h
    bin_y_hi = bin_y_lo + cell_h

    macro_left = hard_pos[:, 0] - hard_sizes[:, 0] / 2
    macro_right = hard_pos[:, 0] + hard_sizes[:, 0] / 2
    macro_bot = hard_pos[:, 1] - hard_sizes[:, 1] / 2
    macro_top = hard_pos[:, 1] + hard_sizes[:, 1] / 2

    ov_x = F.relu(torch.min(macro_right.unsqueeze(1), bin_x_hi.unsqueeze(0))
                  - torch.max(macro_left.unsqueeze(1), bin_x_lo.unsqueeze(0)))
    ov_y = F.relu(torch.min(macro_top.unsqueeze(1), bin_y_hi.unsqueeze(0))
                  - torch.max(macro_bot.unsqueeze(1), bin_y_lo.unsqueeze(0)))

    density = torch.mm(ov_x.T, ov_y) / (cell_w * cell_h)

    total_macro_area = (hard_sizes[:, 0] * hard_sizes[:, 1]).sum()
    target_density = total_macro_area / (canvas_w * canvas_h)

    excess = F.relu(density - target_density * 1.5)
    return (excess ** 2).sum()


def density_loss_top_k(
    positions: torch.Tensor,
    benchmark,
    grid_col: int = 0,
    grid_row: int = 0,
) -> torch.Tensor:
    """
    Density loss matching the evaluator: soft top-K average of grid cell
    densities, including ALL macros (hard + soft) on the benchmark's grid.

    The evaluator computes density_cost = 0.5 * avg(top_10%_cell_densities).
    This function provides a differentiable approximation using softmax-weighted
    average that focuses gradients on the highest-density cells.
    """
    device = positions.device
    num_macros = benchmark.num_macros
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height

    # Use benchmark grid if not specified
    if grid_col == 0:
        grid_col = getattr(benchmark, 'grid_cols', 32)
    if grid_row == 0:
        grid_row = getattr(benchmark, 'grid_rows', 32)

    all_pos = positions[:num_macros]
    all_sizes = benchmark.macro_sizes[:num_macros].to(device)

    cell_w = canvas_w / grid_col
    cell_h = canvas_h / grid_row

    bin_x_lo = torch.arange(grid_col, device=device).float() * cell_w
    bin_x_hi = bin_x_lo + cell_w
    bin_y_lo = torch.arange(grid_row, device=device).float() * cell_h
    bin_y_hi = bin_y_lo + cell_h

    macro_left = all_pos[:, 0] - all_sizes[:, 0] / 2
    macro_right = all_pos[:, 0] + all_sizes[:, 0] / 2
    macro_bot = all_pos[:, 1] - all_sizes[:, 1] / 2
    macro_top = all_pos[:, 1] + all_sizes[:, 1] / 2

    ov_x = F.relu(torch.min(macro_right.unsqueeze(1), bin_x_hi.unsqueeze(0))
                  - torch.max(macro_left.unsqueeze(1), bin_x_lo.unsqueeze(0)))
    ov_y = F.relu(torch.min(macro_top.unsqueeze(1), bin_y_hi.unsqueeze(0))
                  - torch.max(macro_bot.unsqueeze(1), bin_y_lo.unsqueeze(0)))

    # density: [grid_col, grid_row] -> flatten
    density = torch.mm(ov_x.T, ov_y) / (cell_w * cell_h)
    density_flat = density.reshape(-1)

    # Soft top-k: temperature-scaled softmax weights high-density cells
    temp = 10.0
    weights = F.softmax(temp * density_flat, dim=0)
    weighted_mean = (weights * density_flat).sum()

    return weighted_mean


def overlap_loss(
    positions: torch.Tensor,
    benchmark,
) -> torch.Tensor:
    """
    Pairwise overlap penalty for hard macros (fully vectorized).
    """
    device = positions.device
    num_hard = benchmark.num_hard_macros

    if num_hard <= 1:
        return torch.tensor(0.0, device=device)

    hard_pos = positions[:num_hard]
    hard_sizes = benchmark.macro_sizes[:num_hard].to(device)

    half_w = hard_sizes[:, 0] / 2
    half_h = hard_sizes[:, 1] / 2

    dx = hard_pos[:, 0].unsqueeze(1) - hard_pos[:, 0].unsqueeze(0)
    dy = hard_pos[:, 1].unsqueeze(1) - hard_pos[:, 1].unsqueeze(0)

    min_sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0)
    min_sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0)

    ov_x = F.relu(min_sep_x - dx.abs())
    ov_y = F.relu(min_sep_y - dy.abs())
    ov_area = ov_x * ov_y

    mask = ~torch.eye(num_hard, dtype=torch.bool, device=device)
    ov_area = ov_area * mask

    return ov_area.sum() / 2


def congestion_loss(
    positions: torch.Tensor,
    graph,
    benchmark,
    gamma: float = 10.0,
) -> torch.Tensor:
    """
    Differentiable RUDY congestion approximation.

    Distributes routing demand uniformly within each net's smooth bounding box.
    Penalizes grid cells with high accumulated demand using soft top-k.
    This targets the evaluator's congestion metric directly.
    """
    device = positions.device
    num_ports = benchmark.port_positions.size(0)

    grid_col = benchmark.grid_cols
    grid_row = benchmark.grid_rows
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height

    # All node positions (macros + ports)
    if num_ports > 0:
        port_pos = benchmark.port_positions.to(device)
        all_positions = torch.cat([positions, port_pos], dim=0)
    else:
        all_positions = positions

    indices = graph.pin_node_indices.clamp(0, all_positions.size(0) - 1)
    pin_positions = all_positions[indices]  # [num_nets, max_degree, 2]
    mask = graph.net_mask  # [num_nets, max_degree]

    large_val = 1e6
    coords = pin_positions
    mask_2d = mask.unsqueeze(-1).expand_as(coords)

    coords_for_max = coords.masked_fill(~mask_2d, -large_val)
    coords_for_min = coords.masked_fill(~mask_2d, large_val)

    # Smooth bounding boxes per net [num_nets, 2]
    bbox_max = torch.logsumexp(gamma * coords_for_max, dim=1) / gamma
    bbox_min = -torch.logsumexp(-gamma * coords_for_min, dim=1) / gamma

    bbox_w = (bbox_max[:, 0] - bbox_min[:, 0]).clamp(min=1e-3)
    bbox_h = (bbox_max[:, 1] - bbox_min[:, 1]).clamp(min=1e-3)

    # Grid cell boundaries
    cell_w = canvas_w / grid_col
    cell_h = canvas_h / grid_row

    bin_x_lo = torch.arange(grid_col, device=device).float() * cell_w
    bin_x_hi = bin_x_lo + cell_w
    bin_y_lo = torch.arange(grid_row, device=device).float() * cell_h
    bin_y_hi = bin_y_lo + cell_h

    # Overlap between net bboxes and grid cells
    ov_x = F.relu(torch.min(bbox_max[:, 0:1], bin_x_hi.unsqueeze(0))
                  - torch.max(bbox_min[:, 0:1], bin_x_lo.unsqueeze(0)))  # [N, grid_col]
    ov_y = F.relu(torch.min(bbox_max[:, 1:2], bin_y_hi.unsqueeze(0))
                  - torch.max(bbox_min[:, 1:2], bin_y_lo.unsqueeze(0)))  # [N, grid_row]

    # RUDY demand: weight * overlap_area / bbox_area per cell
    bbox_area = (bbox_w * bbox_h).clamp(min=cell_w * cell_h * 0.01)
    net_weights = graph.net_weights
    demand_weight = net_weights / bbox_area  # [N]

    weighted_ov_x = ov_x * demand_weight.unsqueeze(1)  # [N, grid_col]
    demand = torch.mm(weighted_ov_x.T, ov_y)  # [grid_col, grid_row]

    # Normalize by routing capacity
    h_cap = cell_h * benchmark.hroutes_per_micron
    v_cap = cell_w * benchmark.vroutes_per_micron
    avg_cap = (h_cap + v_cap) / 2
    demand_norm = demand / max(avg_cap, 1e-6)

    # Soft top-5% (evaluator uses abu with 5% for congestion)
    demand_flat = demand_norm.reshape(-1)
    temp = 10.0
    weights = F.softmax(temp * demand_flat, dim=0)
    congestion = (weights * demand_flat).sum()

    return congestion


def spreading_loss(
    positions: torch.Tensor,
    benchmark,
) -> torch.Tensor:
    """
    Lightweight spreading force to encourage macros to use the canvas area.

    Penalizes the variance of macro positions being too low (all clustered together).
    """
    device = positions.device
    num_hard = benchmark.num_hard_macros
    hard_pos = positions[:num_hard]
    movable = ~benchmark.macro_fixed[:num_hard].to(device)

    if movable.sum() < 2:
        return torch.tensor(0.0, device=device)

    movable_pos = hard_pos[movable]

    # Normalize to [0, 1] range
    norm_x = movable_pos[:, 0] / benchmark.canvas_width
    norm_y = movable_pos[:, 1] / benchmark.canvas_height

    # Target variance for uniform distribution on [0,1] is 1/12 ~ 0.083
    target_var = 0.06
    loss_x = F.relu(target_var - norm_x.var())
    loss_y = F.relu(target_var - norm_y.var())

    return loss_x + loss_y


def total_loss(
    positions: torch.Tensor,
    graph,
    benchmark,
    epoch: int,
    max_epochs: int = 1000,
    gamma_start: float = 2.0,
    gamma_end: float = 20.0,
) -> tuple:
    """
    Compute total weighted loss with annealing schedule.

    Three phases:
      1. Early (t<0.2): Spreading + basic density + light overlap
      2. Mid (0.2<t<0.6): Ramp up overlap + density_topk + congestion
      3. Late (t>0.6): Strong overlap + density_topk + congestion targeting evaluator
    """
    device = positions.device
    t = epoch / max(max_epochs - 1, 1)

    # Gamma annealing
    gamma = gamma_start + t * (gamma_end - gamma_start)

    # Weight annealing
    w_wl = 1.0
    if t < 0.2:
        w_den = 0.5
        w_ov = 0.1 + t / 0.2 * 0.4
        w_spread = 0.5
    elif t < 0.6:
        w_den = 1.0
        w_ov = 0.5 + (t - 0.2) / 0.4 * 9.5
        w_spread = 0.3
    else:
        w_den = 1.0
        w_ov = 10.0 + (t - 0.6) / 0.4 * 90.0
        w_spread = 0.1

    # density_topk and congestion losses disabled:
    # - density_topk adds 30-40s per GNN restart for 1140-macro benchmarks
    #   without clear benefit (evaluator density better optimized via DenEq)
    # - RUDY congestion doesn't match evaluator's routing-based metric
    w_den_topk = 0.0
    w_cong = 0.0

    # Compute losses
    l_wl = wirelength_loss(positions, graph, benchmark, gamma=gamma)
    l_ov = overlap_loss(positions, benchmark)
    l_den = density_loss(positions, benchmark)
    l_spread = spreading_loss(positions, benchmark)

    # Compute evaluator-aligned losses only when their weights are active
    if w_den_topk > 0:
        l_den_topk = density_loss_top_k(positions, benchmark)
    else:
        l_den_topk = torch.tensor(0.0, device=device)

    if w_cong > 0:
        l_cong = congestion_loss(positions, graph, benchmark, gamma=gamma)
    else:
        l_cong = torch.tensor(0.0, device=device)

    # Normalize
    canvas_area = benchmark.canvas_width * benchmark.canvas_height
    l_wl_norm = l_wl / (canvas_area + 1e-6)
    l_ov_norm = l_ov / (canvas_area + 1e-6)

    total = (w_wl * l_wl_norm + w_den * l_den + w_ov * l_ov_norm +
             w_spread * l_spread + w_den_topk * l_den_topk + w_cong * l_cong)

    loss_dict = {
        'total': total.item(),
        'wirelength': l_wl_norm.item(),
        'density': l_den.item(),
        'density_topk': l_den_topk.item(),
        'congestion': l_cong.item(),
        'overlap': l_ov_norm.item(),
        'spread': l_spread.item(),
        'gamma': gamma,
        'w_ov': w_ov,
    }

    return total, loss_dict
