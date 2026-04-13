"""
Hybrid GNN + Electrostatic Macro Placer

Combines GNN initialization with force-directed/electrostatic optimization:
1. Converts the netlist hypergraph into a star-expanded bipartite graph
2. Uses GNN for quick initial placement (few restarts)
3. Refines via electrostatic force-directed optimization (ePlace/RePlAce-style)
   with repulsive density forces + attractive net forces + iterative spreading
4. Legalizes the continuous output to guarantee zero overlaps
5. Final refinement via simulated annealing and coordinate descent

Usage:
    uv run evaluate submissions/gnn_placer/placer.py -b ibm01
    uv run evaluate submissions/gnn_placer/placer.py --all
"""

import sys
import os
import time
import torch

# Add this directory to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import build_graph
from model import PlacementGNN
from losses import total_loss
from legalize import legalize, coordinate_descent_refine, sa_refine
from eplace import ElectrostaticOptimizer

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
from macro_place.loader import load_benchmark_from_dir, load_benchmark


# ── Configuration ──────────────────────────────────────────────────────────

NUM_RESTARTS = 5           # GNN initialization restarts (reduced — GNN is only for init)
GNN_EPOCHS = 150           # Gradient steps per GNN restart
GNN_HIDDEN_DIM = 48        # GNN hidden dimension
GNN_LAYERS = 4             # Number of GNN message-passing layers
GNN_LR = 5e-3              # Initial learning rate
EPLACE_ITERS = 800         # Electrostatic force-directed iterations
EPLACE_GRID = 64           # Density grid resolution for Poisson solver
CD_SWEEPS = 20             # Coordinate descent sweeps (more sweeps = better refinement)
SA_TIME_LIMIT = 30.0       # Time limit for SA refinement (seconds) — fast numpy SA
TOTAL_TIME_LIMIT = 3300.0  # Total time budget per benchmark (55 min, 5 min buffer)

# Known benchmark directories
IBM_ROOT = "external/MacroPlacement/Testcases/ICCAD04"
NG45_DIRS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def _load_plc(benchmark_name: str):
    """Load a PlacementCost object for the given benchmark."""
    if benchmark_name in NG45_DIRS:
        ng45_dir = NG45_DIRS[benchmark_name]
        netlist = f"{ng45_dir}/netlist.pb.txt"
        plc_file = f"{ng45_dir}/initial.plc"
        _, plc = load_benchmark(netlist, plc_file, name=benchmark_name)
    else:
        benchmark_dir = f"{IBM_ROOT}/{benchmark_name}"
        _, plc = load_benchmark_from_dir(benchmark_dir)
    return plc


class GNNPlacer:
    """
    Hybrid GNN + electrostatic force-directed macro placer.

    Pipeline:
        1. Build graph from netlist
        2. Quick GNN initialization (few restarts)
        3. Electrostatic force-directed optimization (main optimizer)
           - Repulsive density forces via FFT Poisson solver
           - Attractive wirelength forces via LSE HPWL gradient
           - Iterative spreading with dynamic penalty weight
        4. Legalize (greedy displacement)
        5. Fast SA refinement (numpy HPWL)
        6. Coordinate descent refinement (true proxy cost)
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n  GNNPlacer (hybrid): {benchmark.name} "
              f"({benchmark.num_hard_macros} hard, {benchmark.num_soft_macros} soft, "
              f"{benchmark.num_nets} nets) on {device}", flush=True)

        # ── Load PlacementCost for true cost evaluation ────────────────
        try:
            plc = _load_plc(benchmark.name)
        except Exception as e:
            print(f"    Warning: Could not load plc: {e}")
            plc = None

        # ── Step 1: Build graph ────────────────────────────────────────
        t0 = time.time()
        graph = build_graph(benchmark, device)
        print(f"    Graph: {graph.num_macro_nodes} macros, "
              f"{graph.num_net_nodes} nets, {graph.num_port_nodes} ports, "
              f"{graph.edge_index_m2n.size(1)} edges [{time.time()-t0:.2f}s]")

        # ── Step 2: Quick GNN initialization ───────────────────────────
        macro_sizes = benchmark.macro_sizes.to(device)
        fixed_positions = benchmark.macro_positions.to(device)
        fixed_mask = benchmark.macro_fixed.to(device)

        best_gnn_positions = None
        best_gnn_cost = float('inf')

        time_for_gnn = min(TOTAL_TIME_LIMIT * 0.05, 180)  # 5% or 3 min max

        for restart in range(NUM_RESTARTS):
            if time.time() - start_time > time_for_gnn:
                print(f"    GNN init budget reached after {restart} restarts")
                break

            t0 = time.time()
            positions = self._gnn_optimize(
                graph, benchmark, macro_sizes, fixed_positions,
                fixed_mask, device,
            )

            # Quick legalization + cost to pick best restart
            legal = legalize(positions.cpu(), benchmark)
            ov = compute_overlap_metrics(legal, benchmark)
            ov_count = ov['overlap_count']

            if plc is not None and ov_count == 0:
                cost = compute_proxy_cost(legal, benchmark, plc)['proxy_cost']
            else:
                cost = 100.0 + ov_count

            elapsed = time.time() - t0
            if restart == 0 or cost < best_gnn_cost:
                print(f"    GNN init {restart+1}: overlaps={ov_count} "
                      f"cost={'N/A' if cost > 99 else f'{cost:.4f}'} [{elapsed:.1f}s]")

            if cost < best_gnn_cost:
                best_gnn_cost = cost
                # Store PRE-legalization positions for ePlace input
                best_gnn_positions = positions.clone()

        if best_gnn_positions is None:
            best_gnn_positions = benchmark.macro_positions.clone().to(device)

        # Legalize the GNN-only result as a fallback reference
        gnn_legal = legalize(best_gnn_positions.cpu(), benchmark)
        gnn_ov = compute_overlap_metrics(gnn_legal, benchmark)
        if plc is not None and gnn_ov['overlap_count'] == 0:
            gnn_only_cost = compute_proxy_cost(gnn_legal, benchmark, plc)['proxy_cost']
        else:
            gnn_only_cost = best_gnn_cost
        print(f"    Best GNN init: "
              f"cost={'N/A' if gnn_only_cost > 99 else f'{gnn_only_cost:.4f}'}")

        # ── Step 3: Electrostatic force-directed optimization ──────────
        remaining = TOTAL_TIME_LIMIT - (time.time() - start_time)
        eplace_time = min(remaining * 0.15, 450)

        if eplace_time > 10:
            t0 = time.time()
            eplace_opt = ElectrostaticOptimizer(
                benchmark, graph, device, grid_size=EPLACE_GRID,
            )
            eplace_result = eplace_opt.optimize(
                best_gnn_positions,
                num_iters=EPLACE_ITERS,
                time_limit=eplace_time,
            )
            print(f"    ePlace total [{time.time()-t0:.1f}s]")
        else:
            eplace_result = best_gnn_positions

        # ── Step 4: Legalize + fallback comparison ─────────────────────
        eplace_legal = legalize(eplace_result.cpu(), benchmark)
        eplace_ov = compute_overlap_metrics(eplace_legal, benchmark)

        if plc is not None and eplace_ov['overlap_count'] == 0:
            eplace_cost = compute_proxy_cost(eplace_legal, benchmark, plc)['proxy_cost']
        else:
            eplace_cost = 100.0 + eplace_ov['overlap_count']

        # Keep whichever is better: ePlace or GNN-only
        if eplace_cost <= gnn_only_cost:
            legal_positions = eplace_legal
            ov = eplace_ov
            print(f"    ePlace wins: {eplace_cost:.4f} vs GNN-only {gnn_only_cost:.4f}")
        else:
            legal_positions = gnn_legal
            ov = gnn_ov
            print(f"    GNN-only wins: {gnn_only_cost:.4f} vs ePlace {eplace_cost:.4f}")

        if plc is not None and ov['overlap_count'] == 0:
            cost_info = compute_proxy_cost(legal_positions, benchmark, plc)
            print(f"    Post-legal cost: {cost_info['proxy_cost']:.4f}")

        # ── Step 5: Fast SA refinement (HPWL-based, numpy) ────────────
        remaining_time = TOTAL_TIME_LIMIT - (time.time() - start_time)
        if remaining_time > 60 and plc is not None and ov['overlap_count'] == 0:
            sa_time = min(SA_TIME_LIMIT, remaining_time * 0.02)
            t0 = time.time()
            legal_positions = sa_refine(
                legal_positions, benchmark, compute_proxy_cost,
                plc, time_limit=sa_time,
            )
            print(f"    SA refinement done [{time.time()-t0:.1f}s]")

        # ── Step 6: Coordinate descent refinement ──────────────────────
        remaining_time = TOTAL_TIME_LIMIT - (time.time() - start_time)
        if remaining_time > 60 and plc is not None:
            ov2 = compute_overlap_metrics(legal_positions, benchmark)
            if ov2['overlap_count'] == 0:
                cd_time = remaining_time * 0.95
                t0 = time.time()
                legal_positions = coordinate_descent_refine(
                    legal_positions, benchmark, compute_proxy_cost,
                    plc, num_sweeps=CD_SWEEPS, time_limit=cd_time,
                )
                print(f"    Coordinate descent done [{time.time()-t0:.1f}s]")

        total_time = time.time() - start_time
        if plc is not None:
            final = compute_proxy_cost(legal_positions, benchmark, plc)
            print(f"    Final: proxy={final['proxy_cost']:.4f} "
                  f"wl={final['wirelength_cost']:.3f} den={final['density_cost']:.3f} "
                  f"cong={final['congestion_cost']:.3f} overlaps={final['overlap_count']}")
        print(f"    Total time: {total_time:.1f}s")

        return legal_positions

    def _gnn_optimize(
        self,
        graph,
        benchmark,
        macro_sizes,
        fixed_positions,
        fixed_mask,
        device,
    ) -> torch.Tensor:
        """Single GNN optimization run."""
        model = PlacementGNN(
            d_macro=8, d_net=4, d_port=4, d_edge=3,
            d_hidden=GNN_HIDDEN_DIM, n_layers=GNN_LAYERS,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=GNN_EPOCHS, eta_min=1e-5
        )

        best_loss = float('inf')
        best_positions = None

        for epoch in range(GNN_EPOCHS):
            optimizer.zero_grad()

            pred_positions = model(
                graph, benchmark.canvas_width, benchmark.canvas_height, macro_sizes
            )

            # Enforce fixed macro positions
            positions = pred_positions.clone()
            positions[fixed_mask] = fixed_positions[fixed_mask]

            loss, _ = total_loss(
                positions, graph, benchmark,
                epoch=epoch, max_epochs=GNN_EPOCHS,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_positions = positions.detach().clone()

        return best_positions
