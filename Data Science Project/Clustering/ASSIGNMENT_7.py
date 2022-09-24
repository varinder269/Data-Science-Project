"""
NAME: VARINDER SINGH
REGISTRATION NUMBER: B20141
MOBILE NUMBER: 7973329710
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as lsm
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

#reading csv file
df = pd.read_csv('Iris.csv')
df_x =df.copy()
test = df['Species']
orginals = []

# Function for finding the purity score of clustered data:
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = lsm(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum()/np.sum(contingency_matrix)

#giving a number notations to the target class
for i in test:
    if i == 'Iris-setosa':
        orginals.append(0)
    elif i == 'Iris-virginica':
        orginals.append(2)
    else:
        orginals.append(1)

#dropping target attribute
df = df.drop(['Species'], axis=1)
names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print("QUESTION 1\n")

# reducing data from 4 dimensional to 2 dimensional
pca = PCA(n_components=2)
reduced = pd.DataFrame(pca.fit_transform(df), columns=['x1', 'x2'])
val, vec = np.linalg.eig(df.corr().to_numpy())
c = np.linspace(1, 4, 4)

plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(7,5), tight_layout = True)
axs.plot(c, [round(i,3) for i in val],color="red")
plt.xticks(np.arange(min(c), max(c)+1, 1.0))
plt.xlabel('Components')
plt.ylabel('Eigen Values')
plt.title('Eigen Values vs Component')
plt.show()

print("QUESTION 2\n")
K = 3  # given
kmeans = KMeans(n_clusters=K)
kmeans.fit(reduced)
k_pred = kmeans.predict(reduced)
reduced['k_cluster'] = kmeans.labels_
kcentres = kmeans.cluster_centers_

print("PART A\n")
# plotting the scatter plot
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(7,5), tight_layout = True)
axs.scatter(reduced[reduced.columns[0]], reduced[reduced.columns[1]], c=k_pred, cmap='autumn', s=85, marker='*')
axs.scatter([kcentres[i][0] for i in range(K)], [kcentres[i][1] for i in range(K)], c='black', marker='*',s=320,label='cluster centres')
plt.legend()
plt.title('Data spread')
plt.xlabel("principal componant-1", size=16)
plt.ylabel("principal componant-2", size=16)
plt.show()

print("PART B")
print('The distortion measure when k=3 is', round(kmeans.inertia_, 3),"\n")
print("PART C")
print('The purity score for k =3 is', round(purity_score(orginals, k_pred)*100, 3),"%\n")

print("QUESTION 3\n")

reduced = reduced.drop(['k_cluster'], axis=1)
Ks = [2, 3, 4, 5, 6, 7]
l1 = []
kpurity = []
for k in Ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced)
    l1.append(round(kmeans.inertia_, 3))
    kpurity.append(round(purity_score(orginals, kmeans.predict(reduced)), 3))
    print(f'The distortion measure when k={k} is', round(kmeans.inertia_, 3))
    print(f'The purity score for k ={k} is',round(purity_score(orginals, kmeans.predict(reduced))*100, 3),"%")
    print()

plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(7,5), tight_layout = True)
axs.plot(Ks, l1, color='red')
plt.xlabel("Value of K", size=16)
plt.ylabel("Distortion measure", size=16)
plt.show()

print("QUESTION 4\n")
# building gmm
gmm = GaussianMixture(n_components=K, random_state=42).fit(reduced)
gmm_pred = gmm.predict(reduced)
reduced['gmm_cluster'] = gmm_pred
gmmcentres = gmm.means_

print("PART A")
# plotting the scatter plot
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(7,5), tight_layout = True)
axs.scatter(reduced[reduced.columns[0]], reduced[reduced.columns[1]], c=gmm_pred, cmap='autumn', s=85, marker='*')
axs.scatter([gmmcentres[i][0] for i in range(K)], [gmmcentres[i][1] for i in range(K)], c='black', marker='*',s=320,label='cluster centres')
plt.legend()
plt.title('Data spread')
plt.xlabel("principal componant-1", size=16)
plt.ylabel("principal componant-2", size=16)
plt.show()

print("\nPART B")
reduced = reduced.drop(['gmm_cluster'], axis=1)
print('The distortion measure for k = 3 Using GMM is', round(gmm.score(reduced) * len(reduced), 3))

print("\nPART C")
print('The purity score for k =3 Using GMM is', round(purity_score(orginals, gmm_pred)*100, 3),"%")

print("\nQUESTION 5\n")
Ks = [2, 3, 4, 5, 6, 7]
l2 = []
kpurity = []
for k in Ks:
    gmm = GaussianMixture(n_components=k, random_state=42).fit(reduced)
    l2.append(round((gmm.score(reduced) * len(reduced)), 3))
    print(f'The distortion measure when k={k} is', round((gmm.score(reduced) * len(reduced)), 3))
    print(f'The purity score for k ={k} is',round(purity_score(orginals, gmm.predict(reduced))*100, 3),"%")
    print()

# plotting K vs distortion measure
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 1,figsize =(7,5), tight_layout = True)
axs.plot(Ks, l2, color='red')
plt.xlabel("GMM - Value of K", size=16)
plt.ylabel("GMM - Distortion measure", size=16)
plt.show()

print("QUESTION 6\n")
e= [1, 1, 5, 5]
min_samp = [4, 10, 4, 10]
for i in range(4):
    dbscan_model = DBSCAN(eps=e[i], min_samples=min_samp[i]).fit(reduced)
    DBSCAN_predictions = dbscan_model.labels_
    print(f'Purity score for "epsl" {e[i]} and "min_samples" {min_samp[i]} = ',round(purity_score(orginals, DBSCAN_predictions)*100, 3),"%")

    plt.style.use('fivethirtyeight')
    fig, axs = plt.subplots(1, 1, figsize=(7, 5), tight_layout=True)
    axs.scatter(reduced[reduced.columns[0]], reduced[reduced.columns[1]], c=DBSCAN_predictions,cmap='autumn', s=85, marker='*')
    plt.title(f'Data spread for "epsl" {e[i]} and "min_samples" {min_samp[i]}')
    plt.xlabel("principal componant-1", size=16)
    plt.ylabel("principal componant-2", size=16)
    plt.show()

