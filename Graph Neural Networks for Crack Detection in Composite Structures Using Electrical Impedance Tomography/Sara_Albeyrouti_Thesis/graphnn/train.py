import os
import pickle

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from .core_config import DataGenerationConfig
from .model import EIT_GNN_Dataset, EIT_Localization_GNN, hungarian_localization_loss

config = DataGenerationConfig()

def train_eit_localization_model(base_path: str, run_evaluation: bool = True):
    """
    Main function to prepare data, initialize the GNN model, and train it
    for localization with train/val/test splits.
    
    Args:
        base_path: Base directory for the project
        run_evaluation: Whether to run evaluation after training (default: True)
    """
    # Data Loading and Splitting 
    print("Starting Model Training")
    print("\nLoading and splitting dataset")
    full_dataset_path = os.path.join(base_path, config.dataset_file)
    df = pd.read_csv(full_dataset_path)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['crack_count'], random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['crack_count'], random_state=42)
    train_ds = EIT_GNN_Dataset(train_df)
    val_ds = EIT_GNN_Dataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    print(f"   Train samples: {len(train_df):,}, Validation samples: {len(val_df):,}, Test samples: {len(test_df):,}")
    print(f"   Train distribution: {train_df['crack_count'].value_counts().sort_index().tolist()}")
    print(f"   Val distribution: {val_df['crack_count'].value_counts().sort_index().tolist()}")
    print(f"   Test distribution: {test_df['crack_count'].value_counts().sort_index().tolist()}")

    # Model and Optimizer Initializationm
    print("\nInitializing model and optimizer")
    sample = train_ds[0]
    model = EIT_Localization_GNN(
        node_dim=sample.x.shape[1],
        delta_dim=sample.delta_features.shape[1]
    ).to(config.device)
    
    # Print model infos
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config.patience, factor=0.5)
    print(f"   Model initialized on {config.device.upper()}")

    # training Loop
    print("\nStarting training loop...")
    model_output_path = os.path.join(base_path, config.model_dir)
    plot_output_path = os.path.join(base_path, config.plot_dir)
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(plot_output_path, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(config.device)
            optimizer.zero_grad()
            preds = model(batch)
            loss = hungarian_localization_loss(preds, batch.y_coords, batch.y_count)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.device)
                preds_val = model(batch)
                loss = hungarian_localization_loss(preds_val, batch.y_coords, batch.y_count)
                val_loss += loss.item()

        # Calculate averages and update history
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(current_lr)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Progress reporting
        if epoch <= 5 or epoch % 10 == 0:
            print(f"   Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {current_lr:.2e}")

        # Model checkpointing and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_output_path, "best_localization_model.pth"))
            patience_counter = 0
            print(f"   → New best model saved (Val Loss: {best_val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # save 
    print("\nTraining complete, save history")
    
    # Save training history
    history_path = os.path.join(model_output_path, "localization_training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Save test split indices (needed for evaluation)
    test_split_path = os.path.join(base_path, "data", "test_split_indices.pkl")
    os.makedirs(os.path.dirname(test_split_path), exist_ok=True)
    with open(test_split_path, 'wb') as f:
        pickle.dump(test_df.index.tolist(), f)
    
    # Save final model state
    torch.save(model.state_dict(), os.path.join(model_output_path, "final_localization_model.pth"))

    # Generate Training Plots
    print("\nGenerating training plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training and  Validation Loss
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2.5, color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2.5, color='orange')
    ax1.set_title('Model Learning Progress', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    
    best_epoch = np.argmin(history['val_loss'])
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({best_epoch+1})')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate Schedule
    ax2 = axes[1]
    ax2.plot(history['learning_rates'], color='red', linewidth=2)
    ax2.set_title('Learning Rate Schedule', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_path, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Training Summary
    print("summary:")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Final learning rate: {history['learning_rates'][-1]:.2e}")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"Training plots saved to: {plot_output_path}")
    print(f"Model saved to: {model_output_path}")
    print(f"Test split indices saved for evaluation")
    print(f"\n training complete")
    return model, history, test_df


