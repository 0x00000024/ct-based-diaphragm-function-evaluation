import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as pltg
import seaborn as sns
from partd import python
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
# import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

columns= ['x_value','y_value','z_value']
# data path
data = pd.read_csv('/Users/Nubee/Downloads/heart_segmentation_data.csv',usecols=columns)
# pdc= o3d.io.read_point_cloud("/Users/Nubee/Downloads/data.csv")

# data columns
x_value_data = data.iloc[:, 0].values
y_value_data = data.iloc[:, 1].values
z_value_data = data.iloc[:, 2].values

# sort data to avoid plotting problems
# x, y, z = zip(*sorted(zip(x_value_data, y_value_data, z_value_data)))
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)
# data_yz = np.array([y, z])
# data_yz = data_yz.transpose()
#
# polynomial_features = PolynomialFeatures(degree=2)
# x_poly = polynomial_features.fit_transform(x[:, np.newaxis])
#
# model = LinearRegression()
# model.fit(x_poly, data_yz)
# y_poly_pred = model.predict(x_poly)
#
# rmse = np.sqrt(mean_squared_error(data_yz, y_poly_pred))
# r2 = r2_score(data_yz, y_poly_pred)
# print("RMSE:", rmse)
# print("R-squared", r2)

# plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(x, data_yz[:, 0], data_yz[:, 1])
# ax.plot(x, y_poly_pred[:, 0], y_poly_pred[:, 1], color='r')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()
# fig.set_dpi(150)

# fig=pltg.Figure(data=[pltg.Scatter3d(x=x,y=y,z=z, mode='markers')])
# # fig.update_layout(title='Segmented Lung', autosize=False,
# #                   width=500, height=500,
# #                   margin=dict(l=65, r=50, b=65, t=90))
# fig.show()

###########################################################################################################################################
# using DBSCAN Model
# Extract the columns from data
data_x=data.loc[:, ['x_value', 'y_value','z_value']].values
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

dbscan = DBSCAN(eps = 3, min_samples = 6).fit(data_x) # fitting the model
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_ # getting the labels

# no of cluster and noise
no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

# initialize SVM classifier
# clf = svm.SVC(kernel='linear')
# X_train, X_test, y_train, y_test, z_train, z_test  = train_test_split(x_value_data, y_value_data, z_value_data, test_size=0.9, random_state=100)
# # clf = clf.fit(X_train,y_train,z_train)

# use transformer to reshape array back to 2D
# transformer= preprocessing.LabelEncoder()
# new_X_train = np.array(X_train).reshape(-1,1)
# new_y_train = np.array(y_train).reshape(-1,1)
# new_z_train = np.array(z_train).reshape(-1,1)

# use classifier
# clf = clf.fit([new_X_train],[new_y_train],[new_z_train])
# # Generate confusion matrix
# matrix = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
# plt.title('Confusion matrix for our classifier')
# plt.show(matrix)
# plt.show()
# # Plot 1
# plt.scatter(data_x[:, 0], data_x[:,1], c = labels, cmap= "plasma") # plotting the clusters
# plt.xlabel("x_value") # X-axis label
# plt.ylabel("y_value") # Y-axis label
# plt.show() # showing the plot

# Plot 2
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(data_x[:, 0], data_x[:, 1],c = colors) # plotting the clusters
# plt.spring()
# plt.xlabel("x_value") # X-axis label
# plt.ylabel("y_value") # Y-axis label
# plt.show() # showing the plot

# Plot 3
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

# Remove the noise
range_max = len(data_x)
X = np.array([data_x[i] for i in range(0, range_max) if labels[i] != -1])
labels = np.array([labels[i] for i in range(0, range_max) if labels[i] != -1])

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


# Clusters 3D Visualization
# fig=pltg.Figure(data=[pltg.Scatter3d(x=data_x[:, 0],y=data_x[:, 1],z=data_x[:, 2], c=labels, mode='markers', marker={'color':'red'})])
# fig.show()

##############################################################################################################################################
