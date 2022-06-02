import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import plotly.graph_objects as pltg
from scipy.cluster.hierarchy import fcluster
from hdbscan import flat
from sklearn.model_selection import train_test_split


# data
data = pd.read_csv('/Users/Nubee/Downloads/heart_Segmentation_data.csv')
data_x=data.loc[:, ['x_value', 'y_value','z_value']].values
x_train,x_test= train_test_split(data_x, test_size =0.20, random_state=0)

plot_kwds = {'alpha' : 0.25, 's' : 10, 'linewidths':0}
print("gotten here")
# used TSNE to set the learning_rate
# projection = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(data)
print("gotten here")
# projection.shape
# plt.scatter(*projection.T, **plot_kwds)
# plt.show()

# fitting the data to the object
clusterer = hdbscan.HDBSCAN(min_cluster_size=120, gen_min_span_tree=True,min_samples=30, cluster_selection_epsilon=0.5).fit(data)



# 3D visualization
# fig=pltg.Figure(data=[pltg.Scatter3d(x=data_x[:, 0],y=data_x[:, 1],z=data_x[:, 2], c= labels, mode='markers', marker={'color':'red'})])
# fig.show()

core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
labels = clusterer.labels_ # getting the labels

# no of cluster and noise
no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)


# Give the clusters a fixed number of clusters
fix_no_clusters= 4
# choose number of resulting clusters
clusterer_flat= flat.HDBSCAN_flat(data_x, cluster_selection_method='eom', n_clusters=fix_no_clusters, inplace=False)


labels_flat = clusterer_flat.labels_
labels_flat_proba=clusterer_flat.probabilities_

# no of cluster_flat and noise_flat
no_clusters_flat = len(np.unique(labels_flat))
no_noise_flat = np.sum(np.array(labels_flat) == -1, axis=0)

print('Estimated no. of clusters_flat: %d' % no_clusters_flat)
print('Estimated no. of noise points_flat: %d' % no_noise_flat)

# Plot 1
# plt.figure(figsize=(7, 3))
# plt.title(f"Flat clustering for {fix_no_clusters} clusters")
# plt.scatter(data_x[labels_flat >= 0, 0], data_x[labels_flat >= 0, 1], c=labels_flat[labels_flat >= 0], s=5,
#             cmap=plt.cm.jet)
# plt.scatter(data_x[labels_flat < 0, 0], data_x[labels_flat < 0, 1], c='k', s=3, marker='x', alpha=0.2)
# plt.show()
# print(f"Unique labels (-1 for outliers): {np.unique(labels_flat)}")

# # Plot 2
unique_labels = set(labels_flat)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels_flat == k

    xy = data_x[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = data_x[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
plt.title("Estimated number of clusters: %d" % no_clusters_flat)
plt.show()
