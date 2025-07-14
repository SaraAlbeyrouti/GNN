"""
GraphNN: Graph Neural Network for EIT Crack Localization

A package for electrical impedance tomography (EIT) crack detection
and localization using graph neural networks.
"""

# Core configuration
from .core_config import DataGenerationConfig
config = DataGenerationConfig()

# Data generation functions
from .data_generation import (
    make_resistors,
    make_perimeter_nodes,
    choose_electrodes,
    build_conductance_matrix,
    ground_conductance_matrix,
    generate_eit_localization_dataset
)

# Model components
from .model import (
    EIT_GNN_Dataset,
    LocalizationOutput,
    EIT_Localization_GNN,
    hungarian_localization_loss
)

# Training functions
from .train import train_eit_localization_model


__version__ = "1.0.0"

__all__ = [
    # Configuration
    "DataGenerationConfig",
    "config",
    
    # Data generation
    "make_resistors",
    "make_perimeter_nodes", 
    "choose_electrodes",
    "build_conductance_matrix",
    "ground_conductance_matrix",
    "generate_eit_localization_dataset",
    
    # Model components
    "EIT_GNN_Dataset",
    "LocalizationOutput", 
    "EIT_Localization_GNN",
    "hungarian_localization_loss",
    
    # Training
    "train_eit_localization_model",


]