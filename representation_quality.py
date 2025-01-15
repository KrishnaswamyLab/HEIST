import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from tqdm import tqdm


def safe_corrcoef(X_np: np.ndarray, eps: float = 1e-8):
    stds = np.std(X_np, axis=0)
    valid_mask = (stds > eps)
    X_nonzero = X_np[:, valid_mask]
    
    if X_nonzero.shape[1] <= 1:
        return None, valid_mask
    
    corr_matrix = np.corrcoef(X_nonzero, rowvar=False)
    
    return corr_matrix, valid_mask

def representation_quality_metrics(
    X: torch.Tensor,
    n_pairs: int = 10000,
    n_clusters: int = 10,
    device: str = 'cpu'
):
   
    X = X.to(device)

    embedding_variance = X.var(dim=0).mean().item()

    X_np = X.detach().cpu().numpy()  # convert to NumPy
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_np)
    labels = kmeans.labels_
    ch_score = calinski_harabasz_score(X_np, labels)

    corr_matrix, _ = safe_corrcoef(X_np)  # shape = (d, d)
    d = X_np.shape[1]
    import pdb; pdb.set_trace()
    # Extract the off-diagonal entries
    off_diag_indices = np.triu_indices(d, k=1)
    avg_abs_correlation = np.mean(np.abs(corr_matrix[off_diag_indices]))

    return embedding_variance, ch_score, avg_abs_correlation

if __name__ == '__main__':
    files = ["data/sea_graphs_3M_attention_anchor_pe_ranknorm_cross_blending.pt",
             "data/sea_graphs_12M_attention_anchor_pe_ranknorm_cross_blending.pt",
             "data/sea_graphs_cross_contrastive_anchor_pe_ranknorm.pt",
             "data/sea_graphs_cross_contrastive_random_walk_projection.pt",
             "data/sea_graphs_info_nce_anchor_ranknorm.pt",
             ]
    for saved_file in files:
        print(saved_file)
        print("Loading graphs")
        _high_level_graphs = torch.load(saved_file)
        n_clusters = len(torch.unique(_high_level_graphs[0].cell_type.cpu()))
        qual = []
        for i in tqdm(range(len(_high_level_graphs))):
            qual.append(list(representation_quality_metrics(_high_level_graphs[i].X, n_clusters=n_clusters)))
        qual = np.array(qual)
        mean_metrics = qual.mean(0)
        max_metrics = qual.max(0)
        print("Mean")
        print(f"Variance: {mean_metrics[0]}, Correlation: {mean_metrics[2]}, Calinski harabasz: {mean_metrics[1]}")
        print("Max")
        print(f"Variance: {max_metrics[0]}, Correlation: {max_metrics[2]}, Calinski harabasz: {max_metrics[1]}")
        del(_high_level_graphs)
        torch.cuda.empty_cache()
