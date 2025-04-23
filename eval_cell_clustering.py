import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
    
if __name__ == '__main__':
    files = ["data/sea_graphs_model_sea_pe_concat_no_cross.pt",
             "data/space-gm/dfci_graphs_model_sea_pe_concat_no_cross.pt",
             "data/space-gm/charville_graphs_model_sea_pe_concat_no_cross.pt",
             "data/space-gm/upmc_graphs_model_sea_pe_concat_no_cross.pt",
             ]
    for saved_file in files:
        print(saved_file)
        print("Loading graphs")
        _high_level_graphs = torch.load(saved_file)
        NMIs = []
        for i in range(len(_high_level_graphs)):
            X = _high_level_graphs[i].X.cpu().numpy()
            y_true = _high_level_graphs[i].cell_type.cpu().numpy()
            for k in range(5):
                kmeans = KMeans(n_clusters=len(np.unique(y_true)))
                y_pred = kmeans.fit_predict(X)
                NMIs.append(normalized_mutual_info_score(y_true, y_pred))
        NMIs = np.array(NMIs)
        print(f"Max: {NMIs.max()}, Mean: {NMIs.mean()}, Std: {NMIs.std()}")
        del(_high_level_graphs)
        torch.cuda.empty_cache()
