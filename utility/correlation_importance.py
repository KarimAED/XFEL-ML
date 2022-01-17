from collections import defaultdict

import matplotlib
matplotlib.style.use("utility/plotting/styling.mplstyle")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from oldMode2017.setup import get_data

data = get_data()

X = np.append(data[0], data[1], axis=0)

#%%
plt.figure(figsize=(8, 4))
ax = plt.subplot(111)
corr = spearmanr(X).correlation
# Ensure the correlation matrix is symmetric
corr = np.abs((corr + corr.T) / 2)
np.fill_diagonal(corr, 1)
ax.set_ylabel(r"$W(x_i, x_j)$")
# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, ax=ax, leaf_rotation=90, count_sort="ascending", no_labels=True
)
dendro_idx = np.arange(0, len(dendro["ivl"]), 5)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
im = ax.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax.set_xticks(dendro_idx)
ax.set_yticks(dendro_idx)
plt.show()

#%%

cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
