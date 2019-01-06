import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


#######################################
#Funciones usadas para aplicar dbscan #
#######################################

#pca_data= data transformada previamente usando pca
#p= porcentaje de la data que sera usada como la cantidad de vecinos mas cercanos
#   valores aceptados: [0,1]
#Algoritmo adaptado de ref[1]
def get_epsilon(pca_data,p):

    """
    Obtiene un radio sugerido para DBSCAN de forma automatica

    Parametros:
    pca_data= Data transformada usando pca o otro metodo.
    p= porcentaje minimo de la data para que se forme un cluster.

    """

    y= []
    m= []
    sm= []
    k= int(np.round(pca_data.shape[0]*p))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(pca_data)
    distances, indices = nbrs.kneighbors(pca_data)
    # Se obtiene la distancia entre el punto y su ultimo vecino mas cercano
    for i in range(distances.shape[0]):
        y.append(max(distances[i]))

    # Se calcula la pendiente para cada punto. Se conservan las mayores a 0
    #x= np.arange(len(y))
    for i in range(len(y)-1):
        slope= np.abs(y[i+1]-y[i])
        if slope != 0 and slope:
            m.append(slope)
    # Se obtiene el correspondiente valor para epsilon

    limit= np.mean(m)+np.std(m)
    for i in m:
        if i > limit:
            sm.append(i)

    return min(sm)

# recibe un arreglo con los posibles valores del radio
def rad_dbscan_analisys(std_data,rad,p,method="silhouette",plot="nclusters",r=False):
    n=len(rad)
    m=std_data.shape[1]-1
    silhouette_matrix = np.zeros((n,m))
    calisnki_matrix = np.zeros((n,m))
    n_clusters_matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            x = apply_pca(std_data,j+1,r="r2")
            db_labels = apply_dbscan(x,rad[i],p,report= False)
            try:
                if method=="silhouette":
                    silhouette_matrix[i][j]= silhouette_score(x,db_labels,metric='euclidean')
                if method=="calinski":
                    calisnki_matrix[i][j] = calinski_harabaz_score(x, db_labels)
                n_clusters_matrix[i][j] = max(db_labels)+1
            except:
                if method=="silhouette":
                    silhouette_matrix[i][j]= -1
                if method=="calinski":
                    calisnki_matrix[i][j] = 0
                n_clusters_matrix[i][j] = max(db_labels)

    if plot != False:
        if plot == "nclusters":
            annot_type= n_clusters_matrix
        if plot == "score":
            annot_type= True

        plt.figure(figsize=(20,16))
        ax = sns.heatmap(silhouette_matrix,linewidths=0,
                         annot=annot_type, xticklabels = np.arange(1,m+1), yticklabels = rad)
        plt.show()

    if r== True:
        return n_clusters_matrix, silhouette_matrix


#pca_data= data transformada previamente usando pca
#rad= corresponde al parametro epsilon dedbscan
#p= porcentaje de la data que sera usada como la cantidad minima para formar un cluster
#   valores aceptados: [0,1]
#report= Indica si se quiere o no imprmir los valores de los parametros usados, y resultados obtenidos
#      valores (True,False)
def apply_dbscan(pca_data,rad,p,report= True):

    """
    Aplica DBSCAN.

    parametros:
    pca_data= data transformada previamente usando pca
    rad= corresponde al parametro epsilon dedbscan
    p= porcentaje de la data que sera usada como la cantidad minima para formar un cluster
    valores aceptados: [0,1]
    report= Indica si se quiere o no imprmir los valores de los parametros usados, y resultados obtenidos
    valores (True,False)

    """

    n_min = np.round(pca_data.shape[0]*p)
    db = DBSCAN(rad,min_samples=n_min).fit(pca_data)
    if report == True:
        sc= silhouette_score(pca_data,db.labels_,metric='euclidean')
        print("###################################")
        print("#####      DBSCAN report      #####")
        print("###################################")
        print("Min samples= ",n_min)
        print("Epsilon(rad)= ",rad)
        print("Number of clusters= ",max(db.labels_)+1)
        if min(db.labels_) < 0:
            print("Noise= ",db.labels_.tolist().count(-1), " elements")
        else:
            print("Noise= False")
        print("Silhouette score= ",sc)
        print("###################################")

    return db.labels_

########################################
#Funciones usadas para aplicar k_Means #
########################################

#pca_data= data transformada previamente usando pca
#method= metodo con el que se evaluara la cantidad de clusters optima
#        valores=  (silhouette, calinsky)
def get_k_clusters(pca_data,method="silhouette"):

    """
    Obtiene la cantidad de clusters segun el maximo score obtenido

    Parametros:
    pca_data= data transformada previamente usando pca
    method= metodo con el que se evaluara la cantidad de clusters optima
    valores=  (silhouette, calinsky)

    """
    score= []
    k= np.arange(2,10)
    for j in k:
        kmeans= KMeans(n_clusters=j, random_state=5, max_iter=500).fit(pca_data)
        predicted_class= kmeans.predict(pca_data)
        if method == "silhouette":
            score.append(silhouette_score(pca_data,predicted_class,metric='euclidean'))
        if method == "calinski":
            score.append(calinski_harabaz_score(pca_data, predicted_class))
    max_index_score= score.index(np.max(score))
    selected_k= k[max_index_score]
    #used_score= score[max_index_score]
    return selected_k, score

#pca_data= data transformada previamente usando pca
#method= metodo con el que se evaluara la cantidad de clusters optima
#        valores=  (silhouette, calinski)
#report= Indica si se quiere o no imprmir los valores de los parametros usados, y resultados obtenidos
#      valores (True,False)
def apply_kmeans(pca_data,k,method="silhouette",report= True):
    """
    Aplica K-k_Means

    Parametros:
    pca_data= data transformada previamente usando pca
    method= metodo con el que se evaluara la cantidad de clusters optima
        valores=  (silhouette, calinski)
    report= Indica si se quiere o no imprmir los valores de los parametros usados, y resultados obtenidos
        valores (True,False)

    """
    #k,sc= get_k_clusters(pca_data, method)
    kmeans= KMeans(n_clusters=k, random_state=5, max_iter=500).fit(pca_data)
    predicted_cluster= kmeans.predict(pca_data)
    sc= silhouette_score(pca_data,predicted_cluster,metric='euclidean')
    centroids = kmeans.cluster_centers_
    if report == True:
        print("###################################")
        print("#####      KMEANS report      #####")
        print("###################################")
        print("Number of clusters= ",k)
        if method == "silhouette":
            print("Silhouette score= ",sc)
        if method == "calinski":
            print("calinski harabaz score= ",sc)
        print("###################################")

    return predicted_cluster, centroids

################################################
#Funciones usadas para aplicar otros algoritmos#
################################################


def apply_agg(pca_data,k=2,linkage="ward",affinity="euclidean",report=True,method="silhouette"):

    """
    Aplica clustering aglomerativo.

    """

    agg= AgglomerativeClustering(n_clusters=k,linkage=linkage,affinity=affinity)
    labels= agg.fit_predict(pca_data)
    if report == True:
        sc= silhouette_score(pca_data,labels,metric='euclidean')
        print("###################################")
        print("#####      Agg Report      #####")
        print("###################################")
        print("Number of clusters= ",k)
        print("Linkage= ", linkage)
        print("Affinity= ",affinity)
        if method == "silhouette":
            print("Silhouette score= ",sc)
        if method == "calinski":
            print("calinski harabaz score= ",sc)
        print("###################################")
    return labels

def apply_gm(pca_data,n=2,report=True,method="silhouette"):

    """
    Aplica mixturas Gaussianas para encontrar grupos.

    """

    gm= GaussianMixture(n_components=n).fit(pca_data)
    labels= gm.predict(pca_data)
    if report == True:
        sc= silhouette_score(pca_data,labels,metric='euclidean')
        print("###################################")
        print("#####     Gm Report      #####")
        print("###################################")
        print("Number of clusters= ",max(labels)+1)
        if method == "silhouette":
            print("Silhouette score= ",sc)
        if method == "calinski":
            print("calinski harabaz score= ",sc)
        print("###################################")
    return labels


#labels= Lista de etiquetas que entrega algun algoritmo de clustering
def get_cluster_index(labels):

    """
    Obtiene los indices para cada clusters, segun los labels entregados
    por algun algoritmo de clustering.

    Parametros:
    labels= Lista de etiquetas que entrega algun algoritmo de clustering.
    """

    nclusters= max(labels)
    min_labels= min(labels)
    k= np.arange(0,nclusters+1)
    clusters= []
    noise= []
    for j in k:
        cluster_index= []
        for i in range(len(labels)):
            if labels[i] != j:
                cluster_index.append(i)
        clusters.append(cluster_index)
    if min_labels == -1:
        for i in range(len(labels)):
            if labels[i] != -1:
                noise.append(i)
    return clusters,noise


#xpca= data transformada usando PCA
#x= data sin ningun tipo de tratamiento
#xstd= data estandarizada
#cluster_index= indices que identifican a cada cluster
#noise_index= idices que identifican la data ruidosa. Se acepta lista vacia.
def get_clusters(pca_data,data,labels,r="r1"):

    """
    Dos listas para la data transformada con PCA u otro metodo y para la
    data sin normalizar. Cada elemenro de estas listas representa a un cluster distinto.
    """

    clusters_index, noise_index= get_cluster_index(labels)
    pca_data = pca_data.tolist()
    data = data.reset_index(drop=True)
    #xstd = xstd.reset_index(drop=True)
    clusters_pca_data = []
    clusters_data=[]
    for i in range(len(clusters_index)):
        clusters_pca_data.append(np.delete(pca_data,clusters_index[i],0))
        clusters_data.append(data.drop(data.index[clusters_index[i]]))
        #clusters_xstd.append(xstd.drop(xstd.index[clusters_index[i]]))
    if noise_index:
        noisy_pca_data= np.delete(pca_data,noise_index,0)
        noisy_data= data.drop(data.index[noise_index])
    else:
        noisy_pca_data= []
        noisy_data= []
        #noisy_data_xstd= pd.DataFrame([])
    if r=="r1":
        return clusters_pca_data, clusters_data,noisy_pca_data, noisy_data
    if r== "r2":
        return clusters_pca_data, clusters_data

def drop_cluster(pca_data,data):

    """
    Elimina el cluster que posee menos datos. Retorna la data transformada por pca
    u otro metodo y la data no normalizada sin separar por cluster.

    """
    k= []
    for i in range(len(pca_data)):
        k.append(pca_data[i].shape[0])
    index= k.index(min(k))

    new_pca_data= []
    new_data= pd.DataFrame()
    #new_data_std= pd.DataFrame()
    for i in range(len(pca_data)):
        if i !=index:
            for j in range(pca_data[i].shape[0]):
                new_pca_data.append(pca_data[i][j])
            new_data= pd.concat([new_data,data[i]],ignore_index=True)
            #new_data_std= pd.concat([new_data_std,data_std[i]], ignore_index=True)

    return index,np.array(new_pca_data),new_data

def append_clusters(data1,data2):

    """
    Agrega un nuevo cluster a la lista de clusters.

    """
    new_data= []
    new_data.append(data1)
    for i in range(len(data2)):
        new_data.append(data2[i])
    return new_data
