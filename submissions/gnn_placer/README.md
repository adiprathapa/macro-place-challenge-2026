# GNN placer

Hybrid GNN + electrostatic macro placer, my entry for the Partcl x HRT Macro Placement Challenge 2026.

## How it works

1. A graph neural network embeds the netlist graph and produces an initial macro placement.
2. ePlace style density optimization spreads macros with an FFT solved electrostatic field.
3. Density equalization and congestion aware coordinate descent refine the layout.
4. A legalization pass removes any remaining overlap.

## Results

Evaluated on all 17 IBM benchmarks with zero overlapping macros.

## Run

From the repo root:

```
uv run evaluate submissions/gnn_placer/placer.py -b ibm01
```

## Files

| File | Purpose |
| --- | --- |
| `placer.py` | Entry point, orchestrates the full pipeline |
| `graph.py` | Netlist graph construction |
| `model.py` | GNN for placement initialization |
| `eplace.py` | Electrostatic density optimization |
| `losses.py` | Proxy cost and density losses |
| `legalize.py` | Overlap removal and legalization |
