import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
# from pyntcloud import PyntCloud
import pandas as pd
def make_meshgrid(x, y):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


data = pd.read_csv('/root/PycharmProjects/pythonProject/heart_Segmentation_data.csv')
pdata=data.values
# cloud = PyntCloud.from_file('/Users/Nubee/Downloads/heart_index_200_541.ply')
# pdata=cloud.points
# ptarget= data.target
pdata.astype(int)
x_train= pdata[:,1]
x_test= pdata[:,1]
y_train= pdata[:,2]
y_test= pdata[:,2]
z_train=pdata[:,3]
z_test= pdata[:,3]


# generate abnormal novel observations

x_outliers= np.random.randn(20,1)
print('x_train_shape: ', x_train.shape)
print('x_outliers: ', x_outliers)
y_outliers= np.random.randn(20,1)
print('y_train_shape: ', y_train.shape)
print('y_outliers: ', y_outliers)
z_outliers= np.random.randn(20,1)
print('z_train_shape: ', z_train.shape)
print('z_outliers: ', z_outliers)

# fit model
# clf=svm.OneClassSVM(nu=0.50, kernel="rbf", gamma=0.00001)
clf=svm.OneClassSVM(nu=0.50, kernel="rbf", gamma=0.00001)

# fit train
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
clf.fit(x_train)
# predict train
y_pred_train =clf.predict(x_train)
x_test=x_test.reshape(-1,1)
y_pred_test=clf.predict(x_test)
# predict outliers
y_pred_outliers =clf.predict(x_outliers)

# no of error train
n_error_train = y_pred_train[y_pred_train == -1].size
# no of error test
n_error_test = y_pred_test[y_pred_test == -1].size
# no of error outliers
n_error_outliers = y_pred_outliers[y_pred_outliers == -1].size

print("n_error_train : ", n_error_train)
print("n_error_test : ", n_error_test)
print("y_pred_outliers : ",y_pred_outliers)
print("n error outliers : ", n_error_outliers)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = x_train,y_train

print('got to meshgrid')
# xx, yy = make_meshgrid(X0, X1)
xx, yy = make_meshgrid(X0, X1)
xx.shape, yy.shape
print('past meshgrid')
# plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_pred_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
print('got to plot')
plt.show()






