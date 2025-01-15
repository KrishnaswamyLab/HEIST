import torch
from torch_geometric.data import DataLoader, Dataset
import torch_geometric.transforms as T
import torch
import pymetis
from torch_geometric.utils import to_networkx, subgraph, add_self_loops, k_hop_subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import random
import os
 
# Getting all memory using os.popen()

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label

def create_dataloader(graphs, batch_size, device):
    high_level_graph = graphs[0]
    num_nodes = high_level_graph.num_nodes
    transform = T.AddRandomWalkPE(walk_length=2, attr_name='pe')
    high_level_graph = transform(high_level_graph)
    transform = T.AddRandomWalkPE(walk_length=1, attr_name='pe')
    low_level_graphs = [transform(i) for i in graphs[1:]]
    for graph in low_level_graphs:
        if 'weight' not in graph or graph.edge_attr is None:
            # Add default edge weight of 1.0 for each edge if 'weight' is missing
            num_edges = graph.edge_index.size(1)
            graph.edge_attr = torch.ones((num_edges, 1), dtype=torch.float)
    G = to_networkx(high_level_graph, to_undirected=True)

    num_partitions = num_nodes//batch_size
    adjacency_list = [list(G.neighbors(node)) for node in range(G.number_of_nodes())]
    _, node_partitions = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

    # Step 4: Create partitions from node_partitions
    partitions = [[] for _ in range(num_partitions)]
    for node_idx, part_idx in enumerate(node_partitions):
        partitions[part_idx].append(node_idx)

    # Step 5: Create high-level subgraphs and low-level batches for each partition
    partitioned_batches = []

    for i,partition_node_indices in enumerate(partitions):
        # Create high-level subgraph for the current partition
        partition_node_indices = torch.tensor(partition_node_indices, dtype=torch.long)
        edge_index_partition, _ = subgraph(partition_node_indices, high_level_graph.edge_index, relabel_nodes=True)

        high_level_subgraph = Data(
            X=high_level_graph.X[partition_node_indices],
            pe=high_level_graph.pe[partition_node_indices],
            edge_index=edge_index_partition,
            y=high_level_graph.y[partition_node_indices] if high_level_graph.y is not None else None,
            num_nodes=len(partition_node_indices)
        )

        # Collect corresponding low-level graphs for each node in the partition
        corresponding_low_level_graphs = [low_level_graphs[node_idx] for node_idx in partition_node_indices]

        # Create a batch from low-level graphs
        low_level_batch = Batch.from_data_list(corresponding_low_level_graphs)
        if low_level_batch.pe.shape[0] == 2148:
            import pdb; pdb.set_trace()
        # Store high-level subgraph and corresponding low-level batch
        partitioned_batches.append((high_level_subgraph, low_level_batch, partition_node_indices))

    # Step 6: Define DataLoader for batched subgraphs
    batch_size = 1
    dataloader = DataLoader(partitioned_batches, batch_size=batch_size, shuffle=True)

    return dataloader