import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, factorized
from scipy.ndimage import gaussian_filter

from .core_config import DataGenerationConfig

config = DataGenerationConfig()

def make_resistors() -> List[Tuple[str, int, int]]:
    """
    Generates a list of all possible horizontal and vertical resistor locations in the simulation grid. 
    Each resistor is identified by its type ( horizontal or vertical)  and the (row, column) index of its top-left connected node.
    """
    resistors = []
    # Horizontal resistors connect (i, j) to (i, j+1)
    for i in range(config.grid_size_x):
        for j in range(config.grid_size_y - 1):
            resistors.append(('h', i, j))
    # Vertical resistors connect (i, j) to (i+1, j)
    for i in range(config.grid_size_x - 1):
        for j in range(config.grid_size_y):
            resistors.append(('v', i, j))
    return resistors

def make_perimeter_nodes(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    """
    Returns perimeter nodes coordinates in a clockwise order
    """
    perimeter = []
    if num_rows > 0 and num_cols > 0:
        # Top row (left to right)
        perimeter.extend([(0, col) for col in range(num_cols)])
        # Right column (top to bottom, excluding top-right corner which is already added)
        perimeter.extend([(row, num_cols - 1) for row in range(1, num_rows)])
        # Bottom row (right to left, excluding bottom-right corner)
        if num_rows > 1:
            perimeter.extend([(num_rows - 1, col) for col in range(num_cols - 2, -1, -1)])
        # Left column (bottom to top, excluding bottom-left and top-left corners)
        if num_cols > 1:
            perimeter.extend([(row, 0) for row in range(num_rows - 2, 0, -1)])
    return perimeter

def choose_electrodes() -> List[Tuple[int, int]]:
    """
    Selects evenly spaced electrode locations from the grid's perimeter nodes,
    based on the number of electrodes chosen. 
    """
    perimeter_nodes = make_perimeter_nodes(config.grid_size_x, config.grid_size_y)
    if len(perimeter_nodes) < config.num_electrodes:
        raise ValueError(
            f"Cannot place {config.num_electrodes} electrodes on perimeter. "
            f"Perimeter has only {len(perimeter_nodes)} nodes for a {config.grid_size_x}x{config.grid_size_y} grid."
        )

    # Calculate step size to evenly distribute electrodes along the perimeter
    step = len(perimeter_nodes) / config.num_electrodes
    # Select nodes at calculated intervals
    return [perimeter_nodes[int(i * step)] for i in range(config.num_electrodes)]

def build_conductance_matrix(resistors_list, crack_locations=None) -> csr_matrix:
    """
    Builds the sparse conductance matrix (G) for the EIT forward problem.
    This matrix describes the electrical connections and properties of the grid.

    Args:
        resistors_list (List): A list of all potential resistor elements in the grid.
        crack_locations (Dict): A dictionary where keys are (r_type, r, c)
                                of cracked resistors and values are their resistance.
                                Defaults to None(no cracks).
    Returns:
        csr_matrix: The sparse conductance matrix suitable for this linear algebra operation
    """
    total_nodes = config.grid_size_x * config.grid_size_y
    g_matrix = lil_matrix((total_nodes, total_nodes))

    node_map = np.arange(total_nodes).reshape((config.grid_size_x, config.grid_size_y))
    crack_locations = crack_locations or {}
    base_resistance = config.nominal_resistance

    for r_type, r, c in resistors_list:
        resistance = crack_locations.get((r_type, r, c), base_resistance)
        conductance = 1.0 / resistance

        node1 = node_map[r, c]
        node2 = node_map[r, c + 1] if r_type == 'h' else node_map[r + 1, c]

        g_matrix[node1, node2] -= conductance
        g_matrix[node2, node1] -= conductance
        g_matrix[node1, node1] += conductance
        g_matrix[node2, node2] += conductance

    return g_matrix.tocsr()

def ground_conductance_matrix(g_matrix: csr_matrix, electrode_indices: List[int]) -> Tuple[csr_matrix, int]:
    """
    Grounds the conductance matrix by fixing one node to 0V.
    """
    all_node_indices = set(range(g_matrix.shape[0]))
    non_electrode_nodes = sorted(list(all_node_indices - set(electrode_indices)))

    ground_node_idx = non_electrode_nodes[0] if non_electrode_nodes else electrode_indices[0]
    if not non_electrode_nodes:
        print(f"ALL nodes are electrodes or no non-electrode node found. Grounding electrode {ground_node_idx}.")

    g_matrix_lil = g_matrix.tolil()
    g_matrix_lil[ground_node_idx, :] = 0
    g_matrix_lil[:, ground_node_idx] = 0
    g_matrix_lil[ground_node_idx, ground_node_idx] = 1.0

    return g_matrix_lil.tocsr(), ground_node_idx

def generate_eit_localization_dataset(dataset_path: str, samples_per_class: List[int], seed: int) -> pd.DataFrame:
    """
    Generates a synthetic EIT dataset for crack localization with noise. 

    Args:
        dataset_path (str): The file path where the generated CSV dataset will be saved.
        samples_per_class (List[int]): Number of samples to generate for each crack count.
        seed (int): Random seed for reproducibility.
    Returns:
        pd.DataFrame: The generated dataset containing delta_V measurements and crack coordinates.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("Dataset Generation Parameters")
    print(f"Grid Dimensions: {config.grid_size_x}x{config.grid_size_y}")
    print(f"Electrodes: {config.num_electrodes}")
    print(f"Resistance: {config.nominal_resistance} Ohm")
    print(f"Crack Resistance Value: {config.crack_resistance_value} Ohm")
    print(f"Max Cracks per Sample: {config.max_cracks}")
    print(f"Samples per crack class: {samples_per_class}")
    print(f"Noise Standard Deviation: {config.baseline_noise_sd}")
    print(f"Noise Correlation Sigma: {config.noise_correlation_sigma}")
    print(f"Output File: {dataset_path}")

    # Grid Structure and Electrodes
    resistors = make_resistors()
    electrodes_coords = choose_electrodes()

    # Create mapping from 2D coordinates to 1D indices 
    node_map = {(i, j): i * config.grid_size_y + j for i in range(config.grid_size_x) for j in range(config.grid_size_y)}
    electrode_indices = [node_map[e] for e in electrodes_coords]

    # Healthy Baseline Measurements
    g_intact = build_conductance_matrix(resistors)
    g_intact_grounded, ground_node_intact = ground_conductance_matrix(g_intact, electrode_indices)
    solve_intact = factorized(g_intact_grounded)

    num_elecs = config.num_electrodes
    v_intact_all_patterns = []
    for i in range(num_elecs):
        src_idx = electrode_indices[i]
        sink_idx = electrode_indices[(i + num_elecs // 2) % num_elecs]
        I_vector = np.zeros(g_intact.shape[0])
        I_vector[src_idx] = 1.0
        I_vector[sink_idx] = -1.0
        I_vector[ground_node_intact] = 0.0
        v_intact_all_patterns.append(solve_intact(I_vector))

    # Generate the Damaged States
    data_rows, sample_id = [], 0
    stats = {"kirchhoff_failures": 0, "solver_failures": 0}

    for crack_count, n_samples in enumerate(samples_per_class):
        print(f"\nGenerating {n_samples} samples with {crack_count} cracks")
        for _ in range(n_samples):
            cracked_resistors_list = random.sample(resistors, crack_count)
            crack_info_dict = {pos: config.crack_resistance_value for pos in cracked_resistors_list}

            g_damaged = build_conductance_matrix(resistors, crack_info_dict)
            g_damaged_grounded, ground_node_damaged = ground_conductance_matrix(g_damaged, electrode_indices)

            measurements_delta_V = []
            is_sample_valid = True

            # Validate all patterns 
            for pattern_idx in range(num_elecs):
                src_idx = electrode_indices[pattern_idx]
                sink_idx = electrode_indices[(pattern_idx + num_elecs // 2) % num_elecs]
                I_vector_damaged = np.zeros(g_damaged.shape[0])
                I_vector_damaged[src_idx] = 1.0
                I_vector_damaged[sink_idx] = -1.0
                I_vector_damaged[ground_node_damaged] = 0.0
                try:
                    v_damaged = spsolve(g_damaged_grounded, I_vector_damaged)
                    if np.max(np.abs(g_damaged_grounded @ v_damaged - I_vector_damaged)) > config.kirchhoff_tol:
                        stats["kirchhoff_failures"] += 1
                        is_sample_valid = False
                        break
                except Exception:
                    stats["solver_failures"] += 1
                    is_sample_valid = False
                    break

            if not is_sample_valid:
                continue

            # Calculate the delta measurements
            for pattern_idx in range(num_elecs):
                v_intact_current_pattern = v_intact_all_patterns[pattern_idx]
                src_idx = electrode_indices[pattern_idx]
                sink_idx = electrode_indices[(pattern_idx + num_elecs // 2) % num_elecs]
                I_vector_damaged = np.zeros(g_damaged.shape[0])
                I_vector_damaged[src_idx] = 1.0
                I_vector_damaged[sink_idx] = -1.0
                I_vector_damaged[ground_node_damaged] = 0.0
                v_damaged = spsolve(g_damaged_grounded, I_vector_damaged)

                for j in range(num_elecs):
                    p1_idx = electrode_indices[j]
                    p2_idx = electrode_indices[(j + 1) % num_elecs]
                    dv_damaged = v_damaged[p1_idx] - v_damaged[p2_idx]
                    dv_intact = v_intact_current_pattern[p1_idx] - v_intact_current_pattern[p2_idx]
                    measurements_delta_V.append(dv_damaged - dv_intact)

            # Add noise
            noise = gaussian_filter(np.random.normal(0, config.baseline_noise_sd, len(measurements_delta_V)), 
                                   config.noise_correlation_sigma)
            final_measurements = np.array(measurements_delta_V) + noise

            # Convert crack positions to normalized coordinates
            crack_positions_flat = []
            for r_type, r_idx, c_idx in cracked_resistors_list:
                max_r_norm = max(config.grid_size_x - 1, 1)
                max_c_norm = max(config.grid_size_y - 1, 1)
                x_coord = (c_idx + 0.5) / max_c_norm if r_type == 'h' else c_idx / max_c_norm
                y_coord = r_idx / max_r_norm if r_type == 'h' else (r_idx + 0.5) / max_r_norm
                crack_positions_flat.extend([x_coord, y_coord])
            
            # use zeros for unused crack slots
            crack_positions_flat += [0.0] * (2 * config.max_cracks - len(crack_positions_flat))

            data_rows.append([sample_id, crack_count, *final_measurements, *crack_positions_flat])
            sample_id += 1

    # Create DataFrame and save
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    measurement_cols = [f"delta_{i}" for i in range(len(final_measurements))]
    crack_cols = [f"crack_{i}_{ax}" for i in range(config.max_cracks) for ax in ['x', 'y']] 
    df = pd.DataFrame(data_rows, columns=["sample_id", "crack_count", *measurement_cols, *crack_cols]) # identifier + measurements +  coordinates
    df.to_csv(dataset_path, index=False)

    print(f"\nGeneration complete! Total samples: {len(df)}")
    print(f"Simulation Failures: Kirchhoff={stats['kirchhoff_failures']}, Solver={stats['solver_failures']}")
    print(f"Dataset saved to: {dataset_path}")
    return df
