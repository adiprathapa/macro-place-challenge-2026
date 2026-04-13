"""
GNN model for macro placement.

Pure PyTorch implementation (no torch-geometric dependency).
Uses a bipartite message-passing architecture on the star-expanded
netlist graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Compute mean of src values grouped by index.

    Args:
        src: [E, D] source values
        index: [E] group indices (values in [0, dim_size))
        dim_size: number of output groups

    Returns:
        [dim_size, D] mean-aggregated values
    """
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    idx_expanded = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx_expanded, src)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones(index.size(0), 1, device=src.device, dtype=src.dtype))
    return out / count.clamp(min=1)


class BipartiteGNNLayer(nn.Module):
    """
    One layer of bipartite message passing on the star-expanded graph.

    Phase 1: macro/port -> net (aggregate info from connected macros into net nodes)
    Phase 2: net -> macro/port (broadcast net-level info back to macros)
    """

    def __init__(self, d: int = 64, d_edge: int = 3):
        super().__init__()
        # Phase 1: macro -> net messages
        self.msg_m2n = nn.Sequential(
            nn.Linear(2 * d + d_edge, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        # Phase 2: net -> macro messages
        self.msg_n2m = nn.Sequential(
            nn.Linear(2 * d + d_edge, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        # GRU updates with gating
        self.update_net = nn.GRUCell(d, d)
        self.update_macro = nn.GRUCell(d, d)
        self.norm_net = nn.LayerNorm(d)
        self.norm_macro = nn.LayerNorm(d)

    def forward(
        self,
        h_nodes: torch.Tensor,     # [N_nodes, d] (macros + ports)
        h_nets: torch.Tensor,      # [N_nets, d]
        edge_index_m2n: torch.Tensor,  # [2, E] src=node, dst=net
        edge_attr_m2n: torch.Tensor,   # [E, d_edge]
        edge_index_n2m: torch.Tensor,  # [2, E] src=net, dst=node
        edge_attr_n2m: torch.Tensor,   # [E, d_edge]
    ):
        num_nodes = h_nodes.size(0)
        num_nets = h_nets.size(0)

        # Phase 1: macro/port -> net
        if edge_index_m2n.size(1) > 0:
            src_m, dst_n = edge_index_m2n
            msg_input = torch.cat([h_nodes[src_m], h_nets[dst_n], edge_attr_m2n], dim=-1)
            msg = self.msg_m2n(msg_input)
            agg_net = scatter_mean(msg, dst_n, dim_size=num_nets)
            h_nets = self.norm_net(self.update_net(agg_net, h_nets))

        # Phase 2: net -> macro/port
        if edge_index_n2m.size(1) > 0:
            src_n, dst_m = edge_index_n2m
            msg_input = torch.cat([h_nets[src_n], h_nodes[dst_m], edge_attr_n2m], dim=-1)
            msg = self.msg_n2m(msg_input)
            agg_macro = scatter_mean(msg, dst_m, dim_size=num_nodes)
            h_nodes = self.norm_macro(self.update_macro(agg_macro, h_nodes))

        return h_nodes, h_nets


class PlacementHead(nn.Module):
    """
    Output head: maps node embeddings to (x, y) canvas coordinates.
    Uses sigmoid to keep outputs within canvas bounds.
    """

    def __init__(self, d: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )

    def forward(
        self,
        h_macro: torch.Tensor,      # [num_macros, d]
        movable_mask: torch.Tensor,  # [num_macros] bool
        canvas_w: float,
        canvas_h: float,
        macro_sizes: torch.Tensor,   # [num_macros, 2]
    ) -> torch.Tensor:
        """Returns [num_macros, 2] positions (only movable macros are predicted)."""
        raw = self.mlp(h_macro)  # [num_macros, 2]

        # Sigmoid maps to [0, 1], then scale to canvas with per-macro margins
        half_w = macro_sizes[:, 0] / 2
        half_h = macro_sizes[:, 1] / 2
        margin_x_lo = half_w
        margin_y_lo = half_h
        range_x = canvas_w - macro_sizes[:, 0]
        range_y = canvas_h - macro_sizes[:, 1]

        # Clamp ranges to be positive
        range_x = range_x.clamp(min=1.0)
        range_y = range_y.clamp(min=1.0)

        x = torch.sigmoid(raw[:, 0]) * range_x + margin_x_lo
        y = torch.sigmoid(raw[:, 1]) * range_y + margin_y_lo

        positions = torch.stack([x, y], dim=-1)  # [num_macros, 2]
        return positions


class PlacementGNN(nn.Module):
    """
    Full GNN model for macro placement.

    Architecture:
        NodeTypeProjection -> [BipartiteGNNLayer x n_layers] -> PlacementHead
    """

    def __init__(
        self,
        d_macro: int = 8,
        d_net: int = 4,
        d_port: int = 4,
        d_edge: int = 3,
        d_hidden: int = 64,
        n_layers: int = 6,
    ):
        super().__init__()
        self.d_hidden = d_hidden

        # Type-specific projections to common embedding space
        self.proj_macro = nn.Linear(d_macro, d_hidden)
        self.proj_net = nn.Linear(d_net, d_hidden)
        self.proj_port = nn.Linear(d_port, d_hidden)

        # GNN layers
        self.layers = nn.ModuleList([
            BipartiteGNNLayer(d_hidden, d_edge) for _ in range(n_layers)
        ])

        # Output head
        self.head = PlacementHead(d_hidden)

    def forward(self, graph, canvas_w: float, canvas_h: float, macro_sizes: torch.Tensor):
        """
        Forward pass: graph -> macro positions.

        Args:
            graph: PlacementGraph object
            canvas_w: Canvas width in microns
            canvas_h: Canvas height in microns
            macro_sizes: [num_macros, 2] tensor of macro (width, height)

        Returns:
            [num_macros, 2] tensor of predicted center positions
        """
        # Project node features to common embedding space
        h_macro = self.proj_macro(graph.macro_features)   # [num_macros, d]
        h_net = self.proj_net(graph.net_features)          # [num_nets, d]

        if graph.num_port_nodes > 0:
            h_port = self.proj_port(graph.port_features)   # [num_ports, d]
            h_nodes = torch.cat([h_macro, h_port], dim=0)  # [num_macros + num_ports, d]
        else:
            h_nodes = h_macro

        # Message passing
        for layer in self.layers:
            h_nodes, h_net = layer(
                h_nodes, h_net,
                graph.edge_index_m2n, graph.edge_attr_m2n,
                graph.edge_index_n2m, graph.edge_attr_n2m,
            )

        # Extract macro embeddings (drop port embeddings)
        h_macro_out = h_nodes[:graph.num_macro_nodes]

        # Predict positions
        positions = self.head(
            h_macro_out, graph.movable_mask,
            canvas_w, canvas_h, macro_sizes,
        )

        return positions
