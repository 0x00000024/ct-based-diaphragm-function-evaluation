import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as pltg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

columns= ['x_value','y_value','z_value']
# data path
data = pd.read_csv('/Users/Nubee/Downloads/heart_Segmentation_data.csv',usecols=columns)

#############################################################################################################################################
# using DBSCAN Model
# Extract the columns from data
data_x=data.loc[:, ['x_value', 'y_value','z_value']].values
# data_x = data_x.astype('int')
print('Total data points, Dimensionality of data set: ',data_x.shape)
# creating an object of the NearestNeighbors class
neighb = NearestNeighbors(n_neighbors=3)
nbrs=neighb.fit(data_x) # fitting the data to the object
distances,indices=nbrs.kneighbors(data_x) # finding the nearest neighbours
# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.title('k-distance graph')
plt.xlabel('Data points sorted by distance')
plt.ylabel('Epsilon')
plt.show() # showing the plot

dbscan = DBSCAN(eps = 10, min_samples = 6).fit(data_x) # fitting the model
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_ # getting the labels


# no of cluster and noise
no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)
print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

# x_data= data_x[:,0:1]
# y_data= data_x[:,2]
#
# # train dataset data_x
# x_train, x_test, y_train, y_test= train_test_split(x_data,y_data, test_size=0.20, random_state = 42)
# # Initialize classifier
# clf = svm.SVC(kernel='linear')
# clf = clf.fit(x_train,y_train)
# # Use classifier
# clf = clf.fit([x_train],[y_train])
#
# # Generate confusion matrix
# matrix = plot_confusion_matrix(clf, x_test, cmap=plt.cm.Blues, normalize='true')
# plt.title('Confusion matrix for our classifier')
# plt.show(matrix)
# # plot confusion matrix
# plt.show()

# Plot data_x
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

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

plt.title("Estimated number of clusters: %d" % no_clusters)
plt.show()
##############################################################################################################################################
