import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Se arregla la columna que tiene los valores Q2-Q2promedio
def clean(x):
    x = x.replace(";", "")
    return float(x)

# funcion que obtiene los indices de las estrelllas que cumplen con los criterios
def index_criterion(data,criterion):
    if criterion == "lineal":
        selected = np.where((np.abs(data["Q1-Q1promedio"]) > 4 ) & (np.abs(data["C1"]) > 0.02)
                           & ~(np.abs(data["Q2-Q2promedio"]) > 4))
    if criterion == "parabolic":
        selected = np.where((np.abs(data["Q2-Q2promedio"]) > 4 ) & (np.abs(data["C2"]) > 0.02)
                           & ~(np.abs(data["Q1-Q1promedio"]) > 4))
    if criterion == "mix":
        selected = np.where((np.abs(data["Q1-Q1promedio"]) > 4 ) & (np.abs(data["C1"]) > 0.02)
                            & (np.abs(data["Q2-Q2promedio"]) > 4 ) & (np.abs(data["C2"]) > 0.02))
    return selected

def stars_selection(data,features):

    # Se obtienen los indices de las estrellas que cumplen conlos criterios
    lineal = index_criterion(data,"lineal")
    parabolic = index_criterion(data,"parabolic")
    mix = index_criterion(data,"mix")
    # Se obtienen todas las estrellas que cumplen con los criterios
    df_lineal = data.iloc[lineal]
    df_parabolic = data.iloc[parabolic]
    df_mix = data.iloc[mix]
    dataset = pd.concat([df_lineal,df_parabolic,df_mix],ignore_index=True)
    X = dataset[features]
    X_info = dataset[["Star","RA","DEC"]    ]
    #Se estandariza el dataset
    scaler = StandardScaler().fit(dataset[features])
    X_std = pd.DataFrame(scaler.transform(dataset[features]), columns=features)
    X_std_info = dataset[["Star","RA","DEC"]]
    # Se retorna la data estandarizada y la normal
    return X_std, X_std_info, X, X_info


#######################################################
# Funciones usadas para transformar la data con PCA#
#######################################################

#std_data= Data estandarizada
#n= numero de componentes principales
#r= tipo de retorno
#   r1= retorna la data transformada y los pesos de cada feature por componente
#   r2= retorna solo la data transformada
#   r3= retorna el ratio de varianza que explica cada componente
#   r4= retorna el score del modelo
def apply_pca(std_data,n,r="r1"):
    sklearn_pca = PCA(n_components=n,random_state=1)
    xpca = sklearn_pca.fit_transform(std_data)
    if r == "r1":
        return xpca, sklearn_pca.components_
    elif r == "r2":
        return xpca
    elif r == "r3":
        return sklearn_pca.explained_variance_ratio_
    elif r == "r4":
        return sklearn_pca.score(std_data)
    else:
        print("Error, valores admitidos para r= [r1,r2,r3,r2]")

#components= pesos de cada feature por componente
#features= lista que contiene las features usadas en la transformacion
def pc_info(components,features,print_w=True,print_p=True,r="r0"):
    ind= []
    for i in range(components.shape[0]):
        ind.append("PC"+str(i+1))
    pc_info= []
    feature_weights= pd.DataFrame(np.abs(components),columns= features,index= ind)
    for i in ind:
        info= []
        s= sum(feature_weights.loc[i])
        for j in feature_weights.loc[i]:
            info.append(j/s*100)
        pc_info.append(info)

    feature_percentage= pd.DataFrame(pc_info,columns= features ,index= ind)

    if print_p == True:
        print("#################### Percentages per Features #########################")
        print(feature_percentage.T)
        print("\n")
    if print_w ==True:
        print("#################### Weights per Features #############################")
        print(feature_weights.T)

    if r == "r1":
        return feature_percentage.T
    if r == "r2":
        return feature_weights.T
    if r == "r3":
        return feature_percentage.T,feature_weights.T
