import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GraphConv, GATConv, global_mean_pool, global_max_pool
from scipy.optimize import linear_sum_assignment
from typing import Tuple

from .core_config import DataGenerationConfig
from .data_generation import choose_electrodes

config = DataGenerationConfig()

class EIT_GNN_Dataset(Dataset):
    """
    Grid graph dataset with node features for EIT crack localization.
    Each sample has the node features, edge connectivity,
    delta voltage measurements(which are global features), and ground truth crack coordinates.
    """
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.graph = self._create_static_graph()  # Once only as grid doesnt change  
        
        # Columns from the dataset that the GNN will be using  
        self.delta_cols = [col for col in self.df.columns if col.startswith('delta_')]
        self.crack_cols = [f"crack_{i}_{ax}" for i in range(config.max_cracks) for ax in ['x', 'y']]

    def _create_grid_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the grid graph structure (nodes and edges) with the following node features:
        normalized coordinates (x and y) - for spatial context, electrode proximity (min/max distance) - for relationships to measurement points,          and a boundary flag - also for relationships and current flow.
        
        Returns a tuple which has the 
                - x (torch.Tensor): Node feature matrix of shape [num_nodes, num_node_features].
                - edge_index (torch.Tensor): graph connectivity.
        """
        coords = []
        # Get normalized electrode positions for proximity calculation
        electrode_positions_norm = [
            (e[1] / max(config.grid_size_y - 1, 1), e[0] / max(config.grid_size_x - 1, 1))
            for e in choose_electrodes()
        ]

        for i in range(config.grid_size_x):
            for j in range(config.grid_size_y):
                pos_x_norm = j / max(config.grid_size_y - 1, 1)
                pos_y_norm = i / max(config.grid_size_x - 1, 1)

                # Calculate electrode proximity features
                dists = [
                    np.sqrt((pos_x_norm - ex) ** 2 + (pos_y_norm - ey) ** 2)
                    for ex, ey in electrode_positions_norm
                ]
                min_dist = min(dists)
                max_dist = max(dists)

                # Boundary flag
                is_boundary = 1.0 if (
                    i == 0 or
                    i == config.grid_size_x - 1 or
                    j == 0 or
                    j == config.grid_size_y - 1
                ) else 0.0

                coords.append([pos_x_norm, pos_y_norm, min_dist, max_dist, is_boundary])

        # Building the edges (bidirectional)
        edges = []
        idx_map = {
            (r, c): r * config.grid_size_y + c
            for r in range(config.grid_size_x)
            for c in range(config.grid_size_y)
        }
        for r in range(config.grid_size_x):
            for c in range(config.grid_size_y):
                if c < config.grid_size_y - 1:
                    edges.append((idx_map[(r, c)], idx_map[(r, c + 1)]))
                    edges.append((idx_map[(r, c + 1)], idx_map[(r, c)]))
                if r < config.grid_size_x - 1:
                    edges.append((idx_map[(r, c)], idx_map[(r + 1, c)]))
                    edges.append((idx_map[(r + 1, c)], idx_map[(r, c)]))

        x = torch.tensor(coords, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return x, edge_index

    def _create_static_graph(self):
        """Create and cache the static graph structure."""
        self.x, self.edge_index = self._create_grid_graph()
        return {'x': self.x, 'edge_index': self.edge_index}

    def len(self): 
        """returns the total number of samples in the dataset."""
        return len(self.df)

    def get(self, idx: int):
        """
        Retrieves one data sample by index.
        
        Returns:
            Data: PyTorch Geometric Data object with node features, edges, measurements, and targets
        """
        row = self.df.iloc[idx]
        
        # Extract measurements and targets
        delta = torch.tensor(row[self.delta_cols].values, dtype=torch.float32).unsqueeze(0)
        y_coords = torch.tensor(row[self.crack_cols].values, dtype=torch.float32).view(config.max_cracks, 2)
        y_count = torch.tensor(int(row['crack_count']), dtype=torch.long)
        
        return Data(
            x=self.x.clone(),
            edge_index=self.edge_index.clone(),
            delta_features=delta,
            y_coords=y_coords,
            y_count=y_count
        )

class LocalizationOutput(nn.Module):
    """
    Predicts crack (x,y) coords with sigmoid constraint to [0,1].
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, config.max_cracks * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.fc(x)
        coords = torch.sigmoid(raw)
        return coords.view(-1, config.max_cracks, 2)



class EIT_Localization_GNN(nn.Module):
    """
    GNN regression model.
    """
    def __init__(self, node_dim: int, delta_dim: int):
        super().__init__()
        self.node_enc = nn.Linear(node_dim, config.hidden_dim)
        self.delta_enc = nn.Linear(delta_dim, config.hidden_dim)

        self.conv1 = GraphConv(config.hidden_dim, config.hidden_dim)
        self.conv2 = GATConv(config.hidden_dim, config.hidden_dim, heads=4, concat=False)
        self.conv3 = GraphConv(config.hidden_dim, config.hidden_dim)

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(config.hidden_dim) for _ in range(3)
        ])
        
        # LayerNorm layers 
        self.gnn_norm = nn.LayerNorm(config.hidden_dim * 2) 
        self.delta_norm = nn.LayerNorm(config.hidden_dim)
        
        # The fusion dimension 
        fusion_dim = (config.hidden_dim * 2) + config.hidden_dim
        self.out_head = LocalizationOutput(fusion_dim)

    def forward(self, data: Data) -> torch.Tensor:
        h = F.relu(self.node_enc(data.x))

        for i, (conv, bn) in enumerate(zip(
            [self.conv1, self.conv2, self.conv3], self.bns
        )):
            h = conv(h, data.edge_index)
            h = bn(h)
            h = F.relu(h)
            if i < len(self.bns) - 1:
                h = F.dropout(h, p=config.dropout_rate, training=self.training)

        # two types of pooling
        graph_rep_mean = global_mean_pool(h, data.batch)
        graph_rep_max = global_max_pool(h, data.batch)
        graph_rep = torch.cat([graph_rep_mean, graph_rep_max], dim=1)
        
        delta_rep = F.relu(self.delta_enc(data.delta_features.squeeze(1)))
        

        graph_rep = self.gnn_norm(graph_rep)
        delta_rep = self.delta_norm(delta_rep)

        fused = torch.cat([graph_rep, delta_rep], dim=1)
        
        return self.out_head(fused)

def hungarian_localization_loss(preds, targets, counts):
    """
    Compute Hungarian matching loss for multi-crack localization.
    
    Args:
        preds: Predicted coordinates [batch_size, max_cracks, 2]
        targets: Ground truth coordinates from PyG batching
        counts: Number of actual cracks per sample [batch_size]
    """
    batch_size = preds.size(0)
    total_loss = 0.0
    
    # Handle PyTorch Geometric batching - targets come as [batch_size * max_cracks, 2]
    if targets.dim() == 2 and targets.size(0) == batch_size * config.max_cracks:
        targets = targets.view(batch_size, config.max_cracks, 2)
    elif targets.dim() == 2:
        targets = targets.view(batch_size, -1, 2)
    
    # Process each sample in the batch
    for i in range(batch_size):
        n = counts[i].item()
        
        if n == 0:
            # No cracks - all predictions should be near zero
            total_loss += 1 * torch.sum(preds[i] ** 2)
            continue
        
        # Extract predictions and valid targets for this sample
        sample_preds = preds[i]  # [max_cracks, 2]
        valid_targets = targets[i, :n]  # [n, 2]
        
        # Compute cost matrix: distance from each prediction to each valid target
        # We need to handle the shape carefully for cdist
        if n == 1:
            # Special case for single crack to avoid cdist issues
            distances = torch.norm(sample_preds - valid_targets, dim=1)
            cost_matrix = distances.unsqueeze(1)  # [max_cracks, 1]
        else:
            # General case: compute pairwise distances
            cost_matrix = torch.cdist(
                sample_preds.unsqueeze(0),  # [1, max_cracks, 2]
                valid_targets.unsqueeze(0)   # [1, n, 2]
            ).squeeze(0)  # [max_cracks, n]
        
        # Hungarian assignment to find optimal matching
        with torch.no_grad():
            # Convert to numpy for Hungarian algorithm
            cost_np = cost_matrix.cpu().numpy() # no gpu option 
            row_indices, col_indices = linear_sum_assignment(cost_np)
            
            # Only keep assignments for actual targets
            mask = row_indices < n
            row_idx = row_indices[mask]
            col_idx = col_indices[mask]
            
            # Convert back to torch tensors
            row_idx = torch.tensor(row_idx, device=preds.device)
            col_idx = torch.tensor(col_idx, device=preds.device)
        
        # Compute loss on matched pairs
        if len(row_idx) > 0:
            matched_preds = sample_preds[row_idx]  # [n, 2]
            matched_targets = valid_targets[col_idx]  # [n, 2]
            loss = F.mse_loss(matched_preds, matched_targets, reduction='sum')
        else:
            loss = 0.0
        
        # Penalize unmatched predictions (false positives)
        all_pred_indices = torch.arange(config.max_cracks, device=preds.device)
        unmatched_mask = torch.ones(config.max_cracks, dtype=torch.bool, device=preds.device)
        if len(row_idx) > 0:
            unmatched_mask[row_idx] = False
        
        unmatched_preds = sample_preds[unmatched_mask]
        if unmatched_preds.numel() > 0:
            loss += 1 * torch.sum(unmatched_preds ** 2)
        
        # Apply crack-count dependent weighting
        weight = 1.0 + 0.8 * (n - 1) if n > 0 else 1.0
        total_loss += loss * weight

    
    return total_loss / batch_size