# dataset_generator_script.py
#!/usr/bin/env python3

import os
import random
import numpy as np
import pandas as pd

# these are for solving the electrical math really fast
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import factorized, spsolve
# for smoothing our noise to make it more real
from scipy.ndimage import gaussian_filter

import json
from typing import List, Dict, Tuple, Optional, Any

# grab our shared settings
from settings import settings as s


# --- little physics helpers ---

def make_resistors() -> List[Tuple[str, int, int]]:
    """
    generates all the tiny "wires" (resistors) that connect our grid's nodes.
    each wire is either 'h'orizontal or 'v'ertical.
    """
    resistors = []
    # horizontal wires
    for i in range(s.grid_size_x):
        for j in range(s.grid_size_y - 1): # we connect (i,j) to (i,j+1)
            resistors.append(('h', i, j))
    # vertical wires
    for i in range(s.grid_size_x - 1): # we connect (i,j) to (i+1,j)
        for j in range(s.grid_size_y):
            resistors.append(('v', i, j))
    return resistors

def make_perimeter(num_rows: int, num_cols: int) -> List[Tuple[int, int]]:
    """
    figures out all the nodes (junctions) that are on the very edge of our grid,
    listing them nicely in order, like walking around the border.
    """
    perimeter = []
    # top edge (left to right)
    for col in range(num_cols):
        perimeter.append((0, col))
    # right edge (top to bottom, skipping the first corner)
    for row in range(1, num_rows):
        perimeter.append((row, num_cols - 1))
    # bottom edge (right to left, skipping the corner)
    for col in range(num_cols - 2, -1, -1):
        perimeter.append((num_rows - 1, col))
    # left edge (bottom to top, skipping corners)
    for row in range(num_rows - 2, 0, -1):
        perimeter.append((row, 0))
    return perimeter

def choose_electrodes() -> List[Tuple[int, int]]:
    """
    picks out the spots for our electrodes (sensors) evenly around the grid's edge.
    """
    perimeter_nodes = make_perimeter(s.grid_size_x, s.grid_size_y)
    total_perimeter_nodes = len(perimeter_nodes)
    
    # just a quick check to make sure we're not trying to put too many electrodes
    if total_perimeter_nodes < s.num_electrodes:
        raise ValueError(f"can't place {s.num_electrodes} electrodes on a perimeter with only {total_perimeter_nodes} nodes.")
    
    # calculate the average "step" size between each electrode spot
    spacing_step = total_perimeter_nodes / s.num_electrodes
    # figure out the exact spots (indices) on our perimeter list where electrodes go
    indices_for_electrodes = [int(i * spacing_step) for i in range(s.num_electrodes)]

    # grab the actual (row, column) coordinates for those chosen spots
    return [perimeter_nodes[i] for i in indices_for_electrodes]

def crack_coords_to_normalized(crack_info: Tuple[str, int, int]) -> Tuple[float, float]:
    """
    takes a crack's location (like 'h' at row 0, col 0) and turns it into
    a normalized (x, y) coordinate between 0 and 1.
    """
    kind, row_idx, col_idx = crack_info
    num_rows, num_cols = s.grid_size_x, s.grid_size_y
    if kind == 'h': # horizontal crack is halfway across a column, on a specific row
        x_norm = (col_idx + 0.5) / (num_cols - 1) if num_cols > 1 else 0.5
        y_norm = row_idx / (num_rows - 1) if num_rows > 1 else 0.5
    else:  # 'v'ertical crack is halfway down a row, on a specific column
        x_norm = col_idx / (num_cols - 1) if num_cols > 1 else 0.5
        y_norm = (row_idx + 0.5) / (num_rows - 1) if num_rows > 1 else 0.5
    # make sure coordinates stay neatly between 0 and 1
    return np.clip(x_norm, 0, 1), np.clip(y_norm, 0, 1)


# --- simulation and dataset generation ---

def build_the_conductance_matrix(resistors_list: List[Tuple[str, int, int]],
                                 crack_info: Optional[Dict[Tuple[str, int, int], float]] = None) -> csr_matrix:
    """
    builds the big electrical "connection table" (conductance matrix g) for our grid.
    this matrix helps us figure out how current flows.

    it's based on kirchhoff's current law: "what goes in must come out" at each junction.
    
    args:
        resistors_list (list): all the wires (resistors) in our grid.
        crack_info (dict, optional): specific wires that are "broken" (cracked) and their new, higher resistance.
    
    returns:
        csr_matrix: the conductance matrix g, ready for fast electrical calculations.
    """
    total_grid_nodes = s.grid_size_x * s.grid_size_y
    # g is our giant empty "connection table" spreadsheet.
    # we use 'lil_matrix' because it's easy to fill in cell by cell.
    g_matrix = lil_matrix((total_grid_nodes, total_grid_nodes)) 

    # this helps us map grid coordinates (like row 0, col 0) to a simple number (like 0, 1, 2...)
    node_index_map = np.arange(total_grid_nodes).reshape((s.grid_size_x, s.grid_size_y))

    # if no crack info is given, we just pretend it's an empty list
    crack_info = crack_info or {} 

    # now, we go through every single wire (resistor) in our grid
    for r_type, row_idx, col_idx in resistors_list:
        # figure out this wire's resistance: is it broken (cracked) or healthy?
        resistance_value = crack_info.get((r_type, row_idx, col_idx), s.normal_resistance)
        # conductance is just how easily current flows, it's 1 divided by resistance
        conductance_value = 1.0 / resistance_value

        # find the two junctions (nodes) this wire connects
        node1_idx = node_index_map[row_idx, col_idx]
        node2_idx = node_index_map[row_idx, col_idx + 1] if r_type == 'h' else node_index_map[row_idx + 1, col_idx]

        # --- fill out our g matrix (the connection table) based on kirchhoff's law ---
        # for each wire connecting node1 and node2:
        # 1. subtract its conductance from the "cross-connection" spots:
        #    g[node1, node2] and g[node2, node1] represent current flowing *between* them.
        g_matrix[node1_idx, node2_idx] -= conductance_value 
        g_matrix[node2_idx, node1_idx] -= conductance_value

        # 2. add its conductance to the "self-connection" spots:
        #    g[node1, node1] and g[node2, node2] represent the total current flowing *out of* each node
        #    to all its connected wires.
        g_matrix[node1_idx, node1_idx] += conductance_value
        g_matrix[node2_idx, node2_idx] += conductance_value
    
    # finally, we convert our easy-to-fill 'lil_matrix' into 'csr_matrix'.
    # 'csr_matrix' is super fast for doing the actual electrical math (solving for voltages).
    return g_matrix.tocsr()

def ground_the_matrix(g_matrix: csr_matrix, electrode_indices_list: List[int]) -> Tuple[csr_matrix, int]:
    """
    "grounds" our electrical network. this is needed for the math to work properly
    because it gives a reference point (0 volts). we usually pick a non-electrode node.
    """
    all_node_indices = set(range(g_matrix.shape[0]))
    # find nodes that are not electrodes, we'll ground the first one we find
    non_electrode_nodes = sorted(list(all_node_indices - set(electrode_indices_list)))

    if not non_electrode_nodes:
        print("warning: couldn't find a non-electrode node to ground. grounding node 0 instead.")
        ground_node_idx = 0
    else:
        ground_node_idx = non_electrode_nodes[0]

    # make a temporary copy to modify
    grounded_g = g_matrix.tolil()
    # set its row and column to zero (like shorting it to ground)
    grounded_g[ground_node_idx, :] = 0
    grounded_g[:, ground_node_idx] = 0
    # set the diagonal element to 1.0 to make it solvable (like saying it's connected to itself)
    grounded_g[ground_node_idx, ground_node_idx] = 1.0 
    return grounded_g.tocsr(), ground_node_idx

def make_injection_patterns(electrodes_list: List[Tuple[int, int]]) -> Tuple[List, List]:
    """
    defines how we'll inject current and where we'll measure voltage.
    we'll inject current between opposite electrodes and measure voltage between adjacent non-injection electrodes.
    """
    injections, measurement_pairs_list = [], []
    num_electrodes = len(electrodes_list)
    for i in range(num_electrodes):
        # inject current between electrode 'i' and the one opposite it
        src_electrode_idx, sink_electrode_idx = i, (i + num_electrodes // 2) % num_electrodes
        injections.append((electrodes_list[src_electrode_idx], electrodes_list[sink_electrode_idx]))
        
        pairs_for_this_injection = []
        for j in range(num_electrodes):
            # measure voltage between adjacent electrodes 'j' and 'j+1'
            p1_idx, p2_idx = j, (j + 1) % num_electrodes
            # but make sure we're not measuring where we're injecting current!
            if p1_idx in (src_electrode_idx, sink_electrode_idx) or p2_idx in (src_electrode_idx, sink_electrode_idx):
                continue
            pairs_for_this_injection.append((electrodes_list[p1_idx], electrodes_list[p2_idx]))
        measurement_pairs_list.append(pairs_for_this_injection)
    return injections, measurement_pairs_list

def generate_full_dataset( # This function now handles the single dataset generation
    dataset_file_path: str, # where to save the dataset
    samples_per_class_list: List[int], # how many samples for each crack count
    random_seed: int # for consistent random numbers
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    generates a simulated electrical impedance tomography (eit) dataset for crack detection.

    this function models a composite hull as an electrical grid, simulates eit measurements
    under various crack scenarios (0 to max_cracks), adds realistic noise, and performs
    physics validations. the generated data (measurements, crack info) is saved to a csv,
    along with generation statistics and configuration metadata.

    args:
        dataset_file_path (str): full path to save the generated csv dataset.
        samples_per_class_list (list[int]): number of samples for each crack count (0, 1, ..., max_cracks).
        random_seed (int): random seed for reproducible dataset generation.

    returns:
        tuple[pandas.dataframe, dict]:
            - df (pandas.dataframe): the generated dataset.
            - stats (dict): statistics from the generation process.
    """
    
    # set seeds for consistent random numbers for this specific dataset
    random.seed(random_seed)
    np.random.seed(random_seed)

    s.create_directories() # make sure our output folders are ready

    print(f"getting ready to make dataset: {dataset_file_path}")
    print(f"using random seed: {random_seed}")
    print("we're using a 4-probe setup with electrodes evenly spaced.")
  
    all_resistors = make_resistors()
    all_electrodes = choose_electrodes()
    all_injections, all_measurement_pairs = make_injection_patterns(all_electrodes)
    total_measurements_per_sample = sum(len(p) for p in all_measurement_pairs)
    print(f"grid: {s.grid_size_x}x{s.grid_size_y}, electrodes: {s.num_electrodes}, measurements per sample: {total_measurements_per_sample}")

    # map grid coordinates to simple numbers for our math
    node_to_idx_map = {(i, j): i * s.grid_size_y + j for i, j in np.ndindex(s.grid_size_x, s.grid_size_y)}
    electrode_indices_1d = [node_to_idx_map[elec] for elec in all_electrodes]

    # --- simulate the perfect, uncracked material once ---
    g_intact = build_the_conductance_matrix(all_resistors)
    g_intact_solvable, ground_node_idx = ground_the_matrix(g_intact, electrode_indices_1d)
    # pre-solve the intact matrix for speed
    solver_intact = factorized(g_intact_solvable)

    # get all baseline voltages for the intact material
    voltages_intact_all_injections = []
    for src_elec, sink_elec in all_injections:
        current_injection_vector = np.zeros(g_intact.shape[0])
        current_injection_vector[node_to_idx_map[src_elec]] = 1.0
        current_injection_vector[node_to_idx_map[sink_elec]] = -1.0
        current_injection_vector[ground_node_idx] = 0.0 # ensure ground node current is zero
        voltages_intact_all_injections.append(solver_intact(current_injection_vector))

    # set up the columns for our final dataset file
    dataset_columns = ["sample_id",  "crack_count", "signal_magnitude"] + \
                      [f"delta_{i}" for i in range(total_measurements_per_sample)] + \
                      [f"crack_{i}_{ax}" for i in range(s.max_cracks) for ax in ['x', 'y']] + \
                      ["sample_type"] # 0: random, 1: clustered, 2: central

    # keep track of stats and all the generated data rows
    generation_stats = {"kirchhoff_failures": 0, "solver_failures": 0, "max_residuals": []}
    all_data_rows = []
    current_sample_id = 0

    # --- now, generate data for each crack scenario ---
    for crack_num, num_samples_for_class in enumerate(samples_per_class_list): # Changed variable name for clarity
        print(f"\ngenerating {num_samples_for_class} samples with {crack_num} cracks...") # simpler print
        for _ in range(num_samples_for_class): # no tqdm here
            cracks_in_this_sample = []
            this_sample_type = 0 # default to random crack distribution

            if crack_num == 0:
                cracks_in_this_sample = [] # no cracks, just a clean sample
            else:
                random_dist_choice = random.random()

                # 10% chance for cracks to be clustered together
                if random_dist_choice < 0.1: 
                    this_sample_type = 1 # mark as clustered
                    center_of_cluster = random.choice(all_resistors)
                    cx, cy = crack_coords_to_normalized(center_of_cluster)
                    nearby_resistors = [r for r in all_resistors if np.hypot(
                        crack_coords_to_normalized(r)[0] - cx,
                        crack_coords_to_normalized(r)[1] - cy
                    ) < 0.2] # find wires within a certain radius
                    
                    # make sure we get exactly the right number of cracks
                    if len(nearby_resistors) >= crack_num:
                        cracks_in_this_sample = random.sample(nearby_resistors, crack_num)
                    else:
                        cracks_in_this_sample = nearby_resistors.copy()
                        remaining_needed = crack_num - len(cracks_in_this_sample)
                        if remaining_needed > 0:
                            cracks_in_this_sample.extend(random.sample([r for r in all_resistors if r not in cracks_in_this_sample], remaining_needed))
                
                # 40% chance for cracks to be in the center area
                elif random_dist_choice < 0.5: 
                    this_sample_type = 2 # mark as central
                    central_resistors = [
                        r for r in all_resistors
                        if 0.3 < crack_coords_to_normalized(r)[0] < 0.7
                        and 0.3 < crack_coords_to_normalized(r)[1] < 0.7
                    ]
                    
                    if len(central_resistors) >= crack_num:
                        cracks_in_this_sample = random.sample(central_resistors, crack_num)
                    else:
                        cracks_in_this_sample = central_resistors.copy()
                        remaining_needed = crack_num - len(cracks_in_this_sample)
                        if remaining_needed > 0:
                            cracks_in_this_sample.extend(random.sample([r for r in all_resistors if r not in cracks_in_this_sample], remaining_needed))
                # 50% chance for cracks to be completely random
                else: 
                    this_sample_type = 0 # mark as random
                    cracks_in_this_sample = random.sample(all_resistors, crack_num)
            
            # assign how "broken" each cracked wire is (its severity)
            crack_severity_for_this_sample = {c: random.uniform(*s.crack_severity_range) for c in cracks_in_this_sample}
            # build the electrical map for this damaged material
            g_damaged = build_the_conductance_matrix(all_resistors, crack_severity_for_this_sample)
            # ground the damaged matrix for solving
            g_damaged_solvable, ground_node_for_damaged = ground_the_matrix(g_damaged, electrode_indices_1d)

            # --- now, simulate measurements for this damaged material ---
            measurements_for_sample, is_this_sample_valid = [], True
            for inj_idx, (source_elec, sink_elec) in enumerate(all_injections):
                # set up the current injection vector
                current_vector = np.zeros(g_intact.shape[0])
                current_vector[node_to_idx_map[source_elec]] = 1.0
                current_vector[node_to_idx_map[sink_elec]] = -1.0
                current_vector[ground_node_for_damaged] = 0.0

                try:
                    # solve for voltages in the damaged material
                    voltages_damaged = spsolve(g_damaged_solvable, current_vector)
                    # --- check if the physics holds up (kirchhoff's law validation) ---
                    # calculate residual: how far off is g * v from i?
                    residual_error = g_damaged_solvable @ voltages_damaged - current_vector
                    max_residual_error = np.max(np.abs(residual_error))
                    generation_stats["max_residuals"].append(max_residual_error)
                    # if the error is too big, this sample is not valid
                    if max_residual_error > s.kirchhoff_tol:
                        generation_stats["kirchhoff_failures"] += 1
                        is_this_sample_valid = False
                        break # stop processing this sample, it's faulty
                except Exception as e:
                    # if the math solver completely failed, this sample is also not valid
                    generation_stats["solver_failures"] += 1
                    is_this_sample_valid = False
                    break

                voltages_intact = voltages_intact_all_injections[inj_idx] # get baseline voltages
                for meas_elec_c, meas_elec_d in all_measurement_pairs[inj_idx]:
                    # calculate the change in voltage from the intact state
                    delta_voltage = (voltages_damaged[node_to_idx_map[meas_elec_c]] - voltages_damaged[node_to_idx_map[meas_elec_d]]) - \
                                  (voltages_intact[node_to_idx_map[meas_elec_c]] - voltages_intact[node_to_idx_map[meas_elec_d]])
                    measurements_for_sample.append(delta_voltage)

            if not is_this_sample_valid:
                continue # skip this sample if it failed validation

            # --- add realistic noise to our measurements ---
            # 1. generate basic, wobbly noise (its intensity is set by baseline_noise_sd)
            raw_noise = np.random.normal(0, s.baseline_noise_sd, total_measurements_per_sample)
            # 2. "blend" or "smooth" the noise using a gaussian filter (its smoothness by noise_sigma)
            blended_noise = gaussian_filter(raw_noise, s.noise_sigma)
            # 3. add the blended noise to our actual measurements
            measurements_with_noise = np.array(measurements_for_sample) + blended_noise

            # --- optionally, normalize the measurements ---
            # this makes numbers easier for machine learning models to work with
            if s.normalize_measurements:
                mean_meas = np.mean(measurements_with_noise)
                std_meas = np.std(measurements_with_noise)
                # avoid division by zero if std is tiny
                measurements_with_noise = (measurements_with_noise - mean_meas) / (std_meas + 1e-9)

            # calculate a simple magnitude for analysis later
            magnitude = np.sqrt(np.mean(np.square(measurements_with_noise)))

            # store the crack positions, padding with zeros if we have fewer than max_cracks
            normalized_crack_positions = [coord for c in cracks_in_this_sample for coord in crack_coords_to_normalized(c)]
            normalized_crack_positions.extend([0.0] * (2 * s.max_cracks - len(normalized_crack_positions)))

            # add all this sample's info to our list of data rows
            all_data_rows.append([current_sample_id,  crack_num, magnitude] +
                                 measurements_with_noise.tolist() + normalized_crack_positions + [this_sample_type])
            current_sample_id += 1

    # convert our list of data rows into a pandas dataframe
    final_dataset_df = pd.DataFrame(all_data_rows, columns=dataset_columns)
    # save it to a csv file
    final_dataset_df.to_csv(dataset_file_path, index=False)

    # --- save stats and metadata about this generation run ---
    generation_stats["total_samples_generated"] = len(final_dataset_df)
    
    # save stats as a json file
    stats_file_path = os.path.join(s.figure_dir, s.stats_file)
    with open(stats_file_path, 'w') as f:
        json.dump(generation_stats, f, indent=4)

    # save all the settings used for this generation run
    metadata_info = {
        "settings_snapshot": {k: v for k, v in s.__dict__.items()
                           if not k.startswith('__') and not callable(getattr(s, k))},
        "dataset_seed_used": random_seed, # Use the actual seed passed to the function
        "samples_generated_per_class": samples_per_class_list # Use the actual list passed
    }
    # Removed generation time info from metadata
    metadata_file_path = os.path.join(s.figure_dir, s.metadata_file)
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata_info, f, indent=4)

    print(f"\nfinished making the dataset! saved {len(final_dataset_df)} samples to {dataset_file_path}")
    print(f"generation stats saved to: {stats_file_path}")
    # return the dataframe and stats for immediate use if needed (e.g., by an analysis script)
    return final_dataset_df, generation_stats

# --- when this script is run directly, make the dataset! ---
if __name__ == "__main__":
    # This block now correctly calls the defined function: generate_full_dataset

    print("STARTING GENERATION FOR MAIN DATASET")

    
    main_df, main_stats = generate_full_dataset(
        dataset_file_path=s.dataset_save_path, # This is the file path from settings
        samples_per_class_list=s.samples_per_class, # Sample counts from settings
        random_seed=42 # A specific seed for your main dataset
    )
    
    print("\ndataset generation complete.")
