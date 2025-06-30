import os
import random
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import factorized, spsolve
from scipy.ndimage import gaussian_filter
import json
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

# --- Configuration Settings ---
class Settings:
    GRID_SIZE_X = 10
    GRID_SIZE_Y = 10
    NOMINAL_RESISTANCE = 1.0
    NUM_ELECTRODES = 16
    CRACK_SEVERITY_RANGE = (10000, 30000)
    MAX_CRACKS = 3
    SAMPLES_PER_CLASS = [1000, 5000, 6000, 8000]
    BASELINE_NOISE_SD = 5e-5
    NOISE_CORRELATION_SIGMA = 1.5
    NORMALIZE_MEASUREMENTS = True
    KIRCHHOFF_TOL = 1e-3
    DATASET_FILE = "eit_dataset.csv"
    FIGURE_DIR = "figures"
    STATS_FILE = "stats.json"
    METADATA_FILE = "metadata.json"
    
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.FIGURE_DIR, exist_ok=True)

s = Settings

# --- Physics Helpers ---
def make_resistors() -> List[Tuple[str, int, int]]:
    resistors = []
    for i in range(s.GRID_SIZE_X):
        for j in range(s.GRID_SIZE_Y - 1):
            resistors.append(('h', i, j))
    for i in range(s.GRID_SIZE_X - 1):
        for j in range(s.GRID_SIZE_Y):
            resistors.append(('v', i, j))
    return resistors

def make_perimeter(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    perimeter = []
    if num_rows > 0 and num_cols > 0:
        for col in range(num_cols):
            perimeter.append((0, col))
        for row in range(1, num_rows):
            perimeter.append((row, num_cols - 1))
        if num_rows > 1:
            for col in range(num_cols - 2, -1, -1):
                perimeter.append((num_rows - 1, col))
        if num_cols > 1:
            for row in range(num_rows - 2, 0, -1):
                perimeter.append((row, 0))
    return perimeter

def choose_electrodes() -> List[Tuple[int, int]]:
    perimeter_nodes = make_perimeter(s.GRID_SIZE_X, s.GRID_SIZE_Y)
    if len(perimeter_nodes) < s.NUM_ELECTRODES:
        raise ValueError(f"Cannot place {s.NUM_ELECTRODES} electrodes on perimeter")
    step = len(perimeter_nodes) / s.NUM_ELECTRODES
    return [perimeter_nodes[int(i * step)] for i in range(s.NUM_ELECTRODES)]

def crack_coords_to_normalized(crack_info: Tuple[str, int, int]) -> Tuple[float, float]:
    kind, row_idx, col_idx = crack_info
    max_row = max(s.GRID_SIZE_X - 1, 1)
    max_col = max(s.GRID_SIZE_Y - 1, 1)
    
    if kind == 'h':
        x_norm = (col_idx + 0.5) / max_col
        y_norm = row_idx / max_row
    else:  # 'v'
        x_norm = col_idx / max_col
        y_norm = (row_idx + 0.5) / max_row
    return x_norm, y_norm

# --- Core Simulation ---
def build_the_conductance_matrix(
    resistors_list: List[Tuple[str, int, int]],
    crack_info: Optional[Dict[Tuple[str, int, int], float]] = None
) -> csr_matrix:
    total_nodes = s.GRID_SIZE_X * s.GRID_SIZE_Y
    g_matrix = lil_matrix((total_nodes, total_nodes))
    node_index_map = np.arange(total_nodes).reshape((s.GRID_SIZE_X, s.GRID_SIZE_Y))
    crack_info = crack_info or {}

    for r_type, row_idx, col_idx in resistors_list:
        resistance = crack_info.get((r_type, row_idx, col_idx), s.NOMINAL_RESISTANCE)
        conductance = 1.0 / resistance

        node1 = node_index_map[row_idx, col_idx]
        if r_type == 'h':
            node2 = node_index_map[row_idx, col_idx + 1]
        else:
            node2 = node_index_map[row_idx + 1, col_idx]

        g_matrix[node1, node2] -= conductance
        g_matrix[node2, node1] -= conductance
        g_matrix[node1, node1] += conductance
        g_matrix[node2, node2] += conductance

    return g_matrix.tocsr()

def ground_the_matrix(
    g_matrix: csr_matrix,
    electrode_indices_list: List[int]
) -> Tuple[csr_matrix, int]:
    all_nodes = set(range(g_matrix.shape[0]))
    non_electrode_nodes = sorted(list(all_nodes - set(electrode_indices_list)))
    
    if not non_electrode_nodes:
        ground_node_idx = 0
    else:
        ground_node_idx = non_electrode_nodes[0]

    grounded_g = g_matrix.tolil()
    grounded_g[ground_node_idx, :] = 0
    grounded_g[:, ground_node_idx] = 0
    grounded_g[ground_node_idx, ground_node_idx] = 1.0
    return grounded_g.tocsr(), ground_node_idx

def make_injection_patterns(
    electrodes_list: List[Tuple[int, int]]
) -> Tuple[List, List]:
    injections, measurement_pairs = [], []
    num_electrodes = len(electrodes_list)
    
    for i in range(num_electrodes):
        src_idx = i
        sink_idx = (i + num_electrodes // 2) % num_electrodes
        injections.append((electrodes_list[src_idx], electrodes_list[sink_idx]))
        
        pairs = []
        for j in range(num_electrodes):
            p1_idx = j
            p2_idx = (j + 1) % num_electrodes
            if p1_idx not in (src_idx, sink_idx) and p2_idx not in (src_idx, sink_idx):
                pairs.append((electrodes_list[p1_idx], electrodes_list[p2_idx]))
        measurement_pairs.append(pairs)
    
    return injections, measurement_pairs

# --- Dataset Generation ---
def generate_full_dataset(
    dataset_file_path: str,
    samples_per_class_list: List[int],
    random_seed: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    s.create_directories()

    print(f"Generating dataset: {dataset_file_path}")
    print(f"Crack severity: {s.CRACK_SEVERITY_RANGE[0]}-{s.CRACK_SEVERITY_RANGE[1]}Ω")

    resistors = make_resistors()
    electrodes = choose_electrodes()
    injections, measurement_pairs = make_injection_patterns(electrodes)
    total_measurements = sum(len(p) for p in measurement_pairs)
    
    node_map = {(i, j): i * s.GRID_SIZE_Y + j 
                for i in range(s.GRID_SIZE_X) 
                for j in range(s.GRID_SIZE_Y)}
    electrode_indices = [node_map[e] for e in electrodes]

    # Baseline (intact material)
    g_intact = build_the_conductance_matrix(resistors)
    g_intact_grounded, ground_node = ground_the_matrix(g_intact, electrode_indices)
    solve_intact = factorized(g_intact_grounded)
    
    v_intact_all = []
    for src, sink in injections:
        I = np.zeros(g_intact.shape[0])
        I[node_map[src]] = 1.0
        I[node_map[sink]] = -1.0
        I[ground_node] = 0.0
        v_intact_all.append(solve_intact(I))

    # Dataset columns
    columns = ["sample_id", "crack_count", "signal_magnitude"]
    columns += [f"delta_{i}" for i in range(total_measurements)]
    columns += [f"crack_{i}_{ax}" for i in range(s.MAX_CRACKS) for ax in ['x', 'y']]
    columns += ["sample_type"]  # 0=random, 1=clustered, 2=central

    stats = {
        "kirchhoff_failures": 0,
        "solver_failures": 0,
        "max_residuals": [],
        "failed_crack_positions": []
    }
    data_rows = []
    sample_id = 0

    for crack_count, n_samples in enumerate(samples_per_class_list):
        print(f"\nGenerating {n_samples} samples with {crack_count} cracks...")
        for _ in tqdm(range(n_samples), desc=f"Cracks: {crack_count}"):
            cracks = []
            sample_type = 0  # Default: random
            
            if crack_count > 0:
                rand_val = random.random()
                
                # Clustered cracks (10%)
                if rand_val < 0.1:
                    sample_type = 1
                    center = random.choice(resistors)
                    cx, cy = crack_coords_to_normalized(center)
                    nearby = [
                        r for r in resistors 
                        if np.hypot(crack_coords_to_normalized(r)[0]-cx, 
                                    crack_coords_to_normalized(r)[1]-cy) < 0.2
                    ]
                    if len(nearby) >= crack_count:
                        cracks = random.sample(nearby, crack_count)
                    else:
                        cracks = nearby + random.sample(
                            [r for r in resistors if r not in nearby],
                            crack_count - len(nearby)
                        )
                
                # Central cracks (30%)
                elif rand_val < 0.4:
                    sample_type = 2
                    central = [
                        r for r in resistors
                        if 0.3 <= crack_coords_to_normalized(r)[0] <= 0.7
                        and 0.3 <= crack_coords_to_normalized(r)[1] <= 0.7
                    ]
                    if len(central) >= crack_count:
                        cracks = random.sample(central, crack_count)
                    else:
                        cracks = central + random.sample(
                            [r for r in resistors if r not in central],
                            crack_count - len(central)
                        )
                
                # Random cracks (60%)
                else:
                    sample_type = 0
                    cracks = random.sample(resistors, crack_count)
            
            crack_severity = {c: random.uniform(*s.CRACK_SEVERITY_RANGE) for c in cracks}
            g_damaged = build_the_conductance_matrix(resistors, crack_severity)
            g_damaged_grounded, ground_node_damaged = ground_the_matrix(g_damaged, electrode_indices)
            
            measurements = []
            sample_valid = True
            
            for idx, (src, sink) in enumerate(injections):
                I = np.zeros(g_intact.shape[0])
                I[node_map[src]] = 1.0
                I[node_map[sink]] = -1.0
                I[ground_node_damaged] = 0.0
                
                try:
                    v_damaged = spsolve(g_damaged_grounded, I)
                    residual = g_damaged_grounded @ v_damaged - I
                    max_residual = np.max(np.abs(residual))
                    stats["max_residuals"].append(max_residual)
                    
                    if max_residual > s.KIRCHHOFF_TOL:
                        stats["kirchhoff_failures"] += 1
                        sample_valid = False
                        for c in cracks:
                            stats["failed_crack_positions"].append(crack_coords_to_normalized(c))
                        break
                except Exception as e:
                    stats["solver_failures"] += 1
                    sample_valid = False
                    for c in cracks:
                        stats["failed_crack_positions"].append(crack_coords_to_normalized(c))
                    break
                
                v_intact = v_intact_all[idx]
                for p1, p2 in measurement_pairs[idx]:
                    dv_damaged = v_damaged[node_map[p1]] - v_damaged[node_map[p2]]
                    dv_intact = v_intact[node_map[p1]] - v_intact[node_map[p2]]
                    measurements.append(dv_damaged - dv_intact)
            
            if not sample_valid:
                continue
            
            # Add correlated noise
            raw_noise = np.random.normal(0, s.BASELINE_NOISE_SD, total_measurements)
            correlated_noise = gaussian_filter(raw_noise, s.NOISE_CORRELATION_SIGMA)
            measurements = np.array(measurements) + correlated_noise
            
            # Normalize measurements
            if s.NORMALIZE_MEASUREMENTS:
                mean = np.mean(measurements)
                std = np.std(measurements)
                measurements = (measurements - mean) / (std + 1e-9)
            
            magnitude = np.sqrt(np.mean(np.square(measurements)))
            
            # Format crack positions
            crack_positions = []
            for c in cracks:
                crack_positions.extend(crack_coords_to_normalized(c))
            crack_positions += [0.0] * (2 * s.MAX_CRACKS - len(crack_positions))
            
            data_rows.append([sample_id, crack_count, magnitude, 
                              *measurements.tolist(), 
                              *crack_positions, 
                              sample_type])
            sample_id += 1

    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(dataset_file_path, index=False)
    
    stats["total_samples_generated"] = len(df)
    
    # Save stats
    with open(os.path.join(s.FIGURE_DIR, s.STATS_FILE), 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Save metadata
    metadata = {
        k: v for k, v in vars(s).items() 
        if not k.startswith('__') and not callable(v)
    }
    with open(os.path.join(s.FIGURE_DIR, s.METADATA_FILE), 'w') as f:
        json.dump({"config": metadata}, f, indent=4)
    
    print(f"\nGenerated {len(df)} samples")
    print(f"Saved to {dataset_file_path}")
    return df, stats

# --- Main Execution ---
if __name__ == "__main__":
    s.create_directories()
    df, stats = generate_full_dataset(
        dataset_file_path=s.DATASET_FILE,
        samples_per_class_list=s.SAMPLES_PER_CLASS,
        random_seed=42
    )
    print("\n" + "="*60)
    print(f"Dataset generation complete! Samples: {len(df)}")
    print(f"Failures: Kirchhoff={stats['kirchhoff_failures']}, Solver={stats['solver_failures']}")
    print("="*60)