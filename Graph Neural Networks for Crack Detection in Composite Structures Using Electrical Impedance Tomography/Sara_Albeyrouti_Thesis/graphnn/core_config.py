import torch

class DataGenerationConfig:
    # Grid and Physical Parameters
    grid_size_x = 10
    grid_size_y = 10
    num_electrodes = 18
    max_cracks = 3

    # Electrical Properties
    nominal_resistance = 1.0  # Ohms (healthy material)
    crack_resistance_value = 50000.0  # Ohms (damaged material)

    # Dataset Generation
    samples_per_class = [10000, 10000, 10000, 10000]  # [0, 1, 2, 3] cracks
    dataset_file = "data/10x10_grid_dataset.csv"

    # Noise Parameters
    baseline_noise_sd = 1e-6  # Standard deviation of measurement noise
    noise_correlation_sigma = 0.5  # Spatial correlation of noise

    # Solver Tolerances
    kirchhoff_tol = 1e-10  # Tolerance for Kirchhoff's law validation

    # Training Parameters
    hidden_dim = 128
    layers = 3
    dropout_rate = 0.3
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-6
    epochs = 100
    patience = 10

    # Device and Paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = "models"
    plot_dir = "plots"
