"""
Graph construction from netlist hypergraph.

Converts the Benchmark's net_nodes (hypergraph) into a star-expanded
bipartite graph suitable for message-passing GNN layers.

Node types:
  - Macro/port nodes: indices [0, num_macros + num_ports)
  - Virtual net nodes: indices [num_macros + num_ports, ...)

Edges are bidirectional between macro/port nodes and their associated net nodes.
"""

import torch
from dataclasses import dataclass
from typing import List


@dataclass
class PlacementGraph:
    """Star-expanded bipartite graph for GNN placement."""
    # Node features
    macro_features: torch.Tensor    # [num_macros, 8]
    net_features: torch.Tensor      # [num_nets, 4]
    port_features: torch.Tensor     # [num_ports, 4]

    # Edge connectivity (COO format, bidirectional)
    # Phase 1 edges: macro/port -> net
    edge_index_m2n: torch.Tensor    # [2, E1]
    edge_attr_m2n: torch.Tensor     # [E1, 3]
    # Phase 2 edges: net -> macro/port
    edge_index_n2m: torch.Tensor    # [2, E2]
    edge_attr_n2m: torch.Tensor     # [E2, 3]

    # Counts
    num_macro_nodes: int
    num_net_nodes: int
    num_port_nodes: int

    # Batched pin-position data for vectorized wirelength loss
    pin_node_indices: torch.Tensor  # [num_nets, max_degree] - node indices per net
    pin_offsets: torch.Tensor       # [num_nets, max_degree, 2] - pin offsets
    net_mask: torch.Tensor          # [num_nets, max_degree] - True for real pins
    net_weights: torch.Tensor       # [num_nets]

    # Movable mask for macros
    movable_mask: torch.Tensor      # [num_macros] bool
    hard_macro_mask: torch.Tensor   # [num_macros] bool


def build_graph(benchmark, device: torch.device = None) -> PlacementGraph:
    """
    Build a star-expanded bipartite graph from a Benchmark.

    Args:
        benchmark: Benchmark dataclass with netlist data
        device: Target device for tensors

    Returns:
        PlacementGraph with all node features, edges, and batched pin data
    """
    if device is None:
        device = torch.device('cpu')

    num_macros = benchmark.num_macros
    num_hard = benchmark.num_hard_macros
    num_ports = benchmark.port_positions.shape[0]
    num_nets = benchmark.num_nets
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    canvas_area = canvas_w * canvas_h

    # ── Macro node features [num_macros, 8] ────────────────────────────
    sizes = benchmark.macro_sizes  # [N, 2]
    positions = benchmark.macro_positions  # [N, 2]

    w_norm = sizes[:, 0] / canvas_w
    h_norm = sizes[:, 1] / canvas_h
    area_norm = (sizes[:, 0] * sizes[:, 1]) / canvas_area
    aspect_ratio = sizes[:, 0] / (sizes[:, 1] + 1e-6)
    is_fixed = benchmark.macro_fixed.float()
    is_hard = torch.zeros(num_macros)
    is_hard[:num_hard] = 1.0
    init_x_norm = positions[:, 0] / canvas_w
    init_y_norm = positions[:, 1] / canvas_h

    macro_features = torch.stack([
        w_norm, h_norm, area_norm, aspect_ratio,
        is_fixed, is_hard, init_x_norm, init_y_norm
    ], dim=-1).to(device)

    # ── Port node features [num_ports, 4] ──────────────────────────────
    if num_ports > 0:
        port_pos = benchmark.port_positions
        port_features = torch.stack([
            port_pos[:, 0] / canvas_w,
            port_pos[:, 1] / canvas_h,
            torch.ones(num_ports),
            torch.zeros(num_ports),
        ], dim=-1).to(device)
    else:
        port_features = torch.zeros(0, 4, device=device)

    # ── Build edges and net features ───────────────────────────────────
    # For star expansion: each net becomes a virtual node, connected to
    # all macros/ports in that net.

    max_degree = 0
    net_degrees = []
    for net_idx in range(num_nets):
        nodes = benchmark.net_nodes[net_idx]
        deg = len(nodes)
        net_degrees.append(deg)
        if deg > max_degree:
            max_degree = deg

    max_degree_val = max(max_degree, 1)

    # Build net features and edges
    net_feat_list = []
    src_m2n_list = []
    dst_m2n_list = []
    attr_m2n_list = []

    # Batched pin data
    pin_indices_padded = torch.zeros(num_nets, max_degree_val, dtype=torch.long)
    pin_offsets_padded = torch.zeros(num_nets, max_degree_val, 2)
    net_mask_tensor = torch.zeros(num_nets, max_degree_val, dtype=torch.bool)

    for net_idx in range(num_nets):
        nodes = benchmark.net_nodes[net_idx]
        degree = len(nodes)
        weight = benchmark.net_weights[net_idx].item()

        # Compute average pin position for net feature
        avg_x, avg_y = 0.0, 0.0
        for k, node_idx in enumerate(nodes.tolist()):
            if node_idx < num_macros:
                px = positions[node_idx, 0].item()
                py = positions[node_idx, 1].item()
            elif node_idx < num_macros + num_ports:
                port_idx = node_idx - num_macros
                px = benchmark.port_positions[port_idx, 0].item()
                py = benchmark.port_positions[port_idx, 1].item()
            else:
                px, py = canvas_w / 2, canvas_h / 2
            avg_x += px
            avg_y += py

            # Fill batched pin data
            pin_indices_padded[net_idx, k] = node_idx
            net_mask_tensor[net_idx, k] = True

        if degree > 0:
            avg_x /= degree
            avg_y /= degree

        net_feat_list.append([
            degree / max_degree_val,
            weight,
            avg_x / canvas_w,
            avg_y / canvas_h,
        ])

        # Create edges: macro/port node <-> virtual net node
        for node_idx in nodes.tolist():
            src_m2n_list.append(node_idx)
            dst_m2n_list.append(net_idx)
            attr_m2n_list.append([0.0, 0.0, weight])

    net_features = torch.tensor(net_feat_list, dtype=torch.float32, device=device) if net_feat_list else torch.zeros(0, 4, device=device)

    # Build edge tensors
    if src_m2n_list:
        edge_index_m2n = torch.tensor(
            [src_m2n_list, dst_m2n_list], dtype=torch.long, device=device
        )
        edge_attr_m2n = torch.tensor(attr_m2n_list, dtype=torch.float32, device=device)
        # Reverse edges for net -> macro/port
        edge_index_n2m = torch.tensor(
            [dst_m2n_list, src_m2n_list], dtype=torch.long, device=device
        )
        edge_attr_n2m = edge_attr_m2n.clone()
    else:
        edge_index_m2n = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_attr_m2n = torch.zeros(0, 3, dtype=torch.float32, device=device)
        edge_index_n2m = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_attr_n2m = torch.zeros(0, 3, dtype=torch.float32, device=device)

    # Movable and hard macro masks
    movable_mask = benchmark.get_movable_mask().to(device)
    hard_macro_mask = benchmark.get_hard_macro_mask().to(device)

    return PlacementGraph(
        macro_features=macro_features,
        net_features=net_features,
        port_features=port_features,
        edge_index_m2n=edge_index_m2n,
        edge_attr_m2n=edge_attr_m2n,
        edge_index_n2m=edge_index_n2m,
        edge_attr_n2m=edge_attr_n2m,
        num_macro_nodes=num_macros,
        num_net_nodes=num_nets,
        num_port_nodes=num_ports,
        pin_node_indices=pin_indices_padded.to(device),
        pin_offsets=pin_offsets_padded.to(device),
        net_mask=net_mask_tensor.to(device),
        net_weights=benchmark.net_weights.to(device),
        movable_mask=movable_mask,
        hard_macro_mask=hard_macro_mask,
    )
