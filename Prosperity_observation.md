# Prosperity Project — Observation

## Objective

### 0. Evaluation Coverage: Environments, Hardware, and Software Axes
**Goal:** Build a reproducible evaluation framework exploring Prosperity behavior across multiple dimensions using a unified simulation pipeline.

**Hardware Axis (Virtual Hardware Models)**
- Prosperity accelerator with configurable tile sizes (M, K, N dimensions)
- Baseline accelerators: Eyeriss, PTB, LoAS, SATO, Stellar, A100
- Fixed clock frequency (500 MHz) across all designs for fair comparison

**Software Axis (Workloads)**
- SNN models: SCNN, ST-SCNN, SpikeFormer, SpikingBERT, SDT
- Datasets: CIFAR-10, CIFAR-100, ImageNet, SST-2
- Activation sparsity patterns from real SNN inference on above models/datasets

---

## 1. Cycle-Accurate Virtual Hardware Simulation

**Goal:** Obtain reliable, comparable performance and energy metrics under identical abstraction assumptions.

### 1.1 Hardware-Level Metrics
**Performance**
- Total execution cycles (layer-by-layer accumulation)
- End-to-end execution time = cycles × clock period
- Critical path analysis accounting for memory and compute bottlenecks

**Energy / Power Breakdown**
- Compute energy: PE operations, MACs × per-MAC energy (∼0.1 pJ/MAC at 28nm)
- On-chip buffer (SRAM) energy: read/write accesses × per-access energy 
- Off-chip DRAM energy: data movement × per-access energy 
- Total system energy = Compute + SRAM + DRAM

### 1.2 Software-Level Metrics
**Sparsity Characterization**
- **Activation Sparsity (Bit Density):** Ratio of spike-firing neurons to total neurons per layer/timestep
- **Product Density:** Ratio of non-zero (activation × weight) products after applying sparsity filtering
- **Bit-level vs. Value-level Sparsity:** Distinction between binary spike sparsity and weight magnitude sparsity
- **Temporal Variation:** Sparsity fluctuation across timesteps and layers (critical for scheduling decisions)

---

## 2. Design Space Exploration (DSE) Based on Simulation

**Goal:** Identify optimal Prosperity tile configurations (M, K) maximizing performance-per-watt under realistic workloads.

### 2.1 Exploration Parameters
| Parameter | Dimension | Range | Impact |
|-----------|-----------|-------|--------|
| **Tile Size M** | Output neurons (N dimension) | 64–512 | PE parallelism, buffer reuse for outputs |
| **Tile Size K** | Input neurons/channels (K dimension) | 8–128 | Accumulation depth, weight reuse |
| **Tile Size N** | Batch/timesteps (M dimension) | 16–256 | Activation reuse opportunities |

### 2.2 DSE Methodology
1. **Sweep Approach:** For each (tile_M, tile_K) combination:
   - Run cycle-accurate simulator on all models/datasets
   - Measure total cycles, SRAM accesses, DRAM accesses
   - Compute energy using access counts × per-access energy

2. **Fixed Parameters:** All other architectural parameters (PE count, buffer size, frequency) held constant

3. **Output Metrics:**
   - M_dse.csv: Performance/energy sensitivity to M dimension
   - K_dse.csv: Performance/energy sensitivity to K dimension
   - Pareto frontier identification

### 2.3 Expected Outcomes
- Quantitative justification for Prosperity's chosen tile sizes (M=256, K=16)
- Trade-off analysis: parallelism vs. buffer reuse vs. control overhead
- Insights on how product sparsity affects optimal tiling decisions
- Transferable guidelines for extending Prosperity to other SNN accelerators (e.g., LoAS)

---

## 3. Methodology

### 3.1 Simulation Pipeline Architecture

```
Data Input Layer
├── configs.py: Static network architecture definitions
│   └── Layer shapes, dimensions (input/output channels, kernel sizes, batch/timestep counts)
├── data/*.pkl: Binary activation sparsity maps (torch.bool tensors)
│   └── Derived from real SNN inference on reference models
└── networks.py: Network construction engine

      ↓

Network Construction
├── create_network(name, spike_info):
│   1. Load architecture from configs.py
│   2. Instantiate layer objects (Conv2D, FC, LIFNeuron, Attention)
│   3. Load pkl and assign sparse_map to each layer's activation_tensor
│   └── Return: ordered list of layer ops with complete metadata
└── compute_num_OPS(nn):
    └── Calculate total operations, FC ops, LIF ops separately

      ↓

Hardware Simulation Layer
├── accelerator.py (Prosperity):
│   ├── Tiling: Partition M, K, N dimensions based on tile_size parameters
│   ├── Data reuse analysis: Buffer locality estimation
│   ├── Access pattern tracking: SRAM read/write counts per layer
│   ├── Cycle calculation: Compute cycles + memory stall cycles
│   └── Return: total_cycles, memory_access_profile
│
└── baselines.py (Eyeriss, PTB, LoAS, etc.):
    ├── Model-specific tiling and PE scheduling
    ├── Memory hierarchy simulation
    └── Return: total_cycles, memory_access_profile

      ↓

Energy Calculation
├── energy.py:
│   ├── SRAM energy: sram_accesses × per_access_energy (from sram_config.json)
│   ├── DRAM energy: dram_accesses × 100 pJ (from mem_reference.csv)
│   ├── PE energy: operations × per_op_energy
│   └── Return: total_energy
│
└── buffer_cacti.py (optional):
    ├── Call external CACTI tool for area/power breakdown
    └── Generate detailed buffer statistics for Figure 10

      ↓

Result Aggregation & Analysis
├── simulator.py: Main orchestrator
│   ├── Iterate all model/dataset combinations
│   ├── Call accelerator simulation for each combination
│   ├── Aggregate results → time.csv, energy.csv
│   └── Generate performance/energy rankings
│
├── dse_post_process.py: DSE result analysis
│   └── Extract Pareto-optimal (M, K) configurations
│
└── sparsity_visualization.py: Workload characterization
    └── Generate density_analysis.csv (Figure 11)
```

### 3.2 Data Sources & Traceability

| Data Type | Source File | Format | Usage |
|-----------|------------|--------|-------|
| Network architecture | configs.py | Dict[str, Tuple] | Layer instantiation |
| Activation sparsity | data/*.pkl | Dict[str, torch.Tensor(bool)] | Sparse_map assignment |
| Layer parameters | networks.py (Tensor/Layer classes) | Class attributes | Simulation initialization |
| Accelerator specs | accelerator.py / baselines.py | Hardcoded constants | Cycle/energy models |
| Energy parameters | sram_config.json | JSON key-value | Per-access energy lookup |
| Reference results | reference/*.xlsx | Excel spreadsheets | Validation baseline |

### 3.3. Architecture differnece 
