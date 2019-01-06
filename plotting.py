import numpy as np
import seaborn as sns
import pandas as pd
from preprocessing import apply_pca
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

###########################################
#Utilidades para obtener limites y escalas#
##########################################


#data= data de la que se obtienen sus maximos y minimos. Puede estar o no estandarizada.
#feature= feature a la que se le buscan maximos y minimos
def limits(data,feature,mode="clusters"):
    """
    Retorna los valores maximos y minimos para un unico o varios clusters, Segun
    alguna feature.

    parametros:
    data= lista de DataFrames o DataFrame con los datos.
    feature= feature de la cual se quieren extraer sus limites.

    """
    if mode == "clusters":
        lim_min= []
        lim_max= []
        for i in range(len(data)):
            lim_min.append(np.min(data[i][feature]))
            lim_max.append(np.max(data[i][feature]))
        return np.min(lim_min), np.max(lim_max)
    else:
        if mode == "unique":
            lim_min= np.min(data[feature])
            lim_max= np.max(data[feature])
            return lim_min, lim_max
        else:
            print("ERROR, valores validos para mode: (clusters,single)")

#data= datos sin estandarizar separados por cluster
#feature= feature que se usara como referencia para obtener los colores
def data_color(data,feature,mode="clusters"):
    """
    Selecciona valores segun la feature de interes para hacer el coloreo.

    Parametros:
    data= datos sin estandarizar. Pueden estar separados por cluster o no.
    feature= feature que se usara como referencia para obtener los colores
    """

    if mode == "clusters":
        color= []
        for i in range(len(data)):
            color.append(data[i][feature].values)
        return color
    else:
        if mode == "unique":
            return data[feature].values
        else:
            print("ERROR, valores validos para mode: [clusters,unique]")

#data= datos sin estandarizar separados por cluster
#feature= feature que se usara como referencia para obtener los colores
def data_size(data,feature,mode="clusters"):

    """
    Obtiene una proporcion para cada medicion de la feature seleccionada
    segun el valor maximo (x_i/x_max).

    Parametros:
    datos sin estandarizar. Pueden estar separados por cluster o no.
    feature= feature que se usara como referencia para obtener las proporciones.

    """

    if mode == "cluster":
        min_value, max_value = limits(data, feature,"clusters")
        size= []
        for i in range(len(data)):
            size.append(data[i][feature].values/max_value)
        return size
    else:
        if mode == "unique":
            min_value, max_value = limits(data, feature,"unique")
            return data[feature].values/max_value
        else:
            print("ERROR, valores validos para mode: [cluster,unique]")


def apply_log10(data_x,data_y,semilogx,semilogy):

    """
    Aplica escala logaritmica en base 10 a los datos.

    Parametros:
    data_x: data usada para el eje x
    data_y: data usada para el eje y
    """

    if semilogx ==  True:
        xdata= np.log10(data_x)
    else:
        xdata= data_x

    if semilogy == True:
        ydata= np.log10(data_y)
    else:
        ydata= data_y
    return xdata,ydata

#############################################
#Funciones para realizar diversas graficas  #
#############################################

#data= Datos sin ningun tipode procesamiento.
#Features=  features de interes para obtener las correlaciones.
def plotting_correlation_matrix(data,features,s=(16,5),c="YlGnBu"):
    """
    Funcion que grafica la matriz de correlaciones de las diferentes features

    parametros:
    data= data sin ningún tipo de procesamiento.
    features= features de interes para obtener las correlaciones.
    s= tupla que indica el tamaño del grafico.

    """
    fig= plt.figure(figsize=s)
    ax= fig.add_subplot(111)
    corr = data[features].corr()
    ax= sns.heatmap(corr,annot=True,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns, cmap=c)

    plt.title("Matriz de Correlaciones", fontsize = 15)
    plt.show()

#std_data= data normalizada
def plot_pca_explained_variance(std_data):
    """
    Grafica la varianza acumulada segun la cantidad de componentes principales

    parametros
    std_data= data normalizada.

    """
    n= std_data.shape[1]
    var_exp= apply_pca(std_data,n,r="r3")*100

    cum_var_exp= np.cumsum(var_exp)
    x= np.arange(1,n+1)
    fig= plt.figure(figsize=(12,8))
    ax= fig.add_subplot(111)
    ax.bar(x, var_exp, alpha=0.5, align="center",label="Varianza explicada indivdual")
    ax.step(x, cum_var_exp, where="mid",label="Varianza explicada acumulada")

    ax.set_ylabel("Varianza explicada")
    ax.set_xlabel("Componentes principales")
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_title("Varianza explicada en pca",fontsize=20)


#data= lista de DataFrames que contiene la data separada por cluster.
#features= lista de features.
def pairplot(data,features):

    """
    Realiza un pairplot segun cierta cantidad de features.

    Parametros:
    data= lista de DataFrames que contiene la data separada por cluster.
    features= lista de features.

    """
    newDF = pd.DataFrame()
    new_features= []
    for i in range(len(data)):
        oldDF = data[i]
        oldDF["cluster"] = ["cluster "+str(i+1)]*len(data[i])
        newDF= pd.concat([newDF,oldDF], ignore_index= True)

    for i in features:
        #if i == "Teff":
        #    newDF["log(Teff)"]= np.log10(newDF["Teff"]
        if i == "Rad1":
            new_features.append("log(Rad1)")
            newDF["log(Rad1)"]= np.log10(newDF["Rad1"])
        elif i == "logLbol":
            new_features.append("log(L/Lsun)")
            newDF["log(L/Lsun)"]= newDF["logLbol"]
        elif i == "Rad2":
            new_features.append("log(Rad2)")
            newDF["log(Rad2)"]= np.log10(newDF["Rad2"])
        elif i == "M1":
            new_features.append("log(M1)")
            newDF["log(M1)"]= np.log10(newDF["M1"])
        elif i == "M2":
            new_features.append("log(M2)")
            newDF["log(M2)"]= np.log10(newDF["M2"])
        elif i == "pm":
            new_features.append("log(pm)")
            newDF["log(pm)"]= np.log10(newDF["pm"])
        elif i == "D (pc)":
            new_features.append("log(D)")
            newDF["log(D)"]= np.log10(newDF["D (pc)"])
        else:
            new_features.append(i)
    sns.pairplot(newDF, hue="cluster",vars= new_features,
    diag_kind="kde", diag_kws=dict(shade=True))


def plotting_boxplots_c(data,features,s=(16,16)):

    """
    Realiza un grafico por feature. Cada grafico tiene un boxplot
    por cluster.

    parametros:
    data= Lista de DataFrames, donde cada DataFrame representa a un cluster.
    features= Lista de features que se usaran para hacer los graficos.

    """

    newDF= pd.DataFrame()
    for i in range(len(data)):
        data_aux = data[i]
        data_aux["clusters"] = ["cluster "+str(i+1)]*len(data[i])
        newDF= pd.concat([newDF,data_aux], ignore_index= True)

    fig = plt.figure(figsize=s)
    m= int(np.ceil(len(features)/2))
    gs = gridspec.GridSpec(m, 2)

    for i in range(m):
        if 2*i <= len(features)-1:
            ax= fig.add_subplot(gs[i, 0])
            ax= sns.boxplot(x="clusters", y=features[2*i],data=newDF,showmeans=True)
            if features[2*i] == "Abs Mag":
                ax.invert_yaxis()
            if (features[2*i] == "Teff" or features[2*i] == "Rad2"
                or features[2*i] == "pm" or features[2*i] == "M1"
                or features[2*i] == "D (pc)" or features[2*i] == "M2"
                or features[2*i] == "Rad1"):
                ax.semilogy()
        if 2*i+1 <= len(features)-1:
            ax= fig.add_subplot(gs[i, 1])
            ax= sns.boxplot(x="clusters", y=features[2*i+1],data=newDF,showmeans=True)
            if features[2*i+1] == "Abs Mag":
                ax.invert_yaxis()
            if (features[2*i+1] == "Teff" or features[2*i+1] == "Rad2"
                or features[2*i+1] == "pm" or features[2*i+1] == "M1"
                or features[2*i+1] == "D (pc)" or features[2*i+1] == "M2"
                or features[2*i+1] == "Rad1"):
                ax.semilogy()


#data= lista de arreglos que contiene la data separada por cluster
#y transformada (PCA, Factor analisys, etc.)
def plot_clusters(data):

    """
    Grafica cada cluster encontrado.

    Parametros:
    data= lista de arreglos que contiene la data separada por cluster

    """

    dim= data[0].shape[1]
    if dim == 2:
        fig = plt.figure(figsize=(12,8))
        ax= fig.add_subplot(111)
        ax.grid(True)
        for i in range(len(data)):
            legend= "cluster "+str(i+1)
            ax.scatter(data[i][:,0],data[i][:,1],c="C"+str(i),label= legend)
            plt.legend()
    elif dim == 3:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection = '3d')
        ax.grid(True)
        for i in range(len(data)):
            legend= "cluster "+str(i+1)
            ax.scatter(data[i][:,0],data[i][:,1],data[i][:,2],c="C"+str(i),
            label= legend)
            plt.legend()
    else:
        print("Error, se tienen mas de 3 dimensiones")


def plot_hr_diagram(data,feature_x="Teff",feature_y="logLbol",size_feature="Rad2",color_map="hot",semilogx=False,semilogy=False,s=(22,10)):

    """
    Realiza un diagrama HR

    Parametros:
    data= Data sin ningun tipo de procesamiento.
    feature_x= feature del eje x.
    feature_y= feature del eje y.
    size_feature= feature referencial para el tamaño de cada punto.
    """

    color= np.log10(data_color(data,"Teff",mode="unique"))
    size= data_size(data,size_feature,mode="unique")
    lmin,lmax= limits(data,feature_y,"unique")

    xdata, ydata= apply_log10(data[feature_x],data[feature_y],semilogx,semilogy)

    fig= plt.figure(figsize=s)
    ax= fig.add_subplot(111)
    ax.scatter(xdata,ydata,c=color,cmap=color_map,marker=".",alpha=0.85,s=10000*size,linewidths=2)
    points= ax.scatter(xdata,ydata,c=color,cmap=color_map,marker=".")
    cbar= fig.colorbar(points,ax=ax)
    #Referencia al sol
    #ax.scatter(5778,0,marker="*",c="g",s=50)
    ax.grid()
    ax.set_facecolor("gainsboro")
    ax.set_title("Diagrama H-R",fontsize=20)
    ax.set_ylim(lmin-0.1*lmin,lmax*1.2)
    if feature_y == "logLbol":
        ax.set_ylabel("Log(L/Lsun)",fontsize=15)
    else:
        ax.set_ylabel(feature_y,fontsize=15)
    if semilogx == True:
        ax.set_xlabel("Log("+feature_x+")",fontsize=15)
    else:
        ax.set_xlabel(feature_x+" [K]",fontsize=15)

    if feature_y == "Abs Mag" or feature_y == "Ftot" or feature_y == "Mean mag":
        ax.invert_yaxis()
    ax.invert_xaxis()


def plot_hr_diagram_c(df,data,feature_x="Teff",feature_y="Abs Mag",size_feature="Rad2"):

    """
    Grafica el diagrama HR separado por cluster.

    parametros:
    df= Toda la data sin ningun tipo de procesamiento.
    data= Lista de DataFrames, donde cada DataFrame representa un cluster.
    """

    size= data_size(data,size_feature,mode="cluster")
    size_df= data_size(df,size_feature,mode="unique")
    lmin,lmax= limits(data,feature_y,"cluster")
    colors=["r","y","m","b","g","c","k"]
    fig = plt.figure(figsize=(16,9))
    ax= fig.add_subplot(111)
    ax.scatter(df[feature_x],df[feature_y],marker="o",c="gray",alpha=0.55,s=1000*size_df)
    for i in range(len(data)):
        mfx= np.round(data[i][feature_x].mean(),3)
        mfy= np.round(data[i][feature_y].mean(),3)
        mfs= np.round(data[i][size_feature].mean(),3)
        l="Average "+feature_x+"= "+str(mfx)+"\n Average "+feature_y+"= "+str(mfy)+"\n Average "+size_feature+"= "+str(mfs)
        ax.scatter(data[i][feature_x],data[i][feature_y],marker="o",label=l,c="C"+str(i),alpha=0.55,s=1000*size[i],linewidths=2)

    ax.semilogx()
    ax.legend()
    ax.grid()
    ax.set_facecolor("gainsboro")
    ax.set_title("H-R Diagram",fontsize=20)
    ax.set_xlabel(feature_x+" [K]",fontsize=15)
    ax.set_ylabel(feature_y,fontsize=15)

    if feature_y == "Abs Mag" or feature_y == "Ftot" or feature_y == "Mean mag":
        ax.invert_yaxis()
    ax.invert_xaxis()


def plot_compare_clusters(df,data,feature_x,feature_y,s=(16,16),semilogx=False,semilogy=False,invert_x=False,invert_y=False):
    """
    Funcion que grafica cada uno de los clusters por separado segun dos Features
    a eleccion.

    Parametros:
    df= Toda la data sin ningun tipo de procesamiento.
    data= Lista de DataFrames, donde cada DataFrame representa a un cluster.
    Feature_x= Feature del eje x.
    Feature_y= Feature del eje y.

    """
    fig = plt.figure(figsize=s)
    m= int(np.ceil(len(data)/2))
    gs = gridspec.GridSpec(m, 2)


    for i in range(m):
        xdf, ydf= apply_log10(df[feature_x],df[feature_y],semilogx,semilogy)
        if 2*i <= len(data)-1:
            mfx_true= np.round(data[2*i][feature_x].mean(),3)
            mfy_true= np.round(data[2*i][feature_y].mean(),3)
            xdata_par, ydata_par= apply_log10(data[2*i][feature_x],data[2*i][feature_y],semilogx,semilogy)
            ax= fig.add_subplot(gs[i, 0])

            mfx= np.round(xdata_par.mean(),3)
            mfy= np.round(ydata_par.mean(),3)
            l= "Average "+feature_x+"= "+str(mfx_true)+"\nAverage "+feature_y+"= "+str(mfy_true)
            ax.scatter(xdf,ydf,c="gray",label="All data", alpha= 0.15)
            ax.scatter(xdata_par,ydata_par,c= "C"+str(2*i),label=l)
            ax.plot(mfx,mfy,c="k",marker= "*")
            ax.legend()
            ax.set_title("Cluster " + str(2*i+1))
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            if invert_x == True:
                ax.invert_xaxis()
            if invert_y == True:
                ax.invert_yaxis()
            ax.grid()

        if 2*i + 1 <= len(data)-1:

            ax= fig.add_subplot(gs[i, 1])
            mfx_true= np.round(data[2*i+1][feature_x].mean(),3)
            mfy_true= np.round(data[2*i+1][feature_y].mean(),3)
            xdata_impar, ydata_impar= apply_log10(data[2*i+1][feature_x],data[2*i+1][feature_y],semilogx,semilogy)

            mfx= np.round(xdata_impar.mean(),3)
            mfy= np.round(ydata_impar.mean(),3)
            l= "Average "+feature_x+"= "+str(mfx_true)+"\nAverage "+feature_y+"= "+str(mfy_true)
            ax.scatter(xdf,ydf,c="gray",label="All data", alpha= 0.15)
            ax.scatter(xdata_impar,ydata_impar,c= "C"+str(2*i+1),label= l)
            ax.plot(mfx,mfy,c="k",marker= "*")
            ax.legend()
            ax.set_title("Cluster " + str(2*i+2))
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            if invert_x == True:
                ax.invert_xaxis()
            if invert_y == True:
                ax.invert_yaxis()
            ax.grid()

    plt.tight_layout()


def plot(df,data,feature_x,feature_y,semilogx=False,semilogy=False,invert_x=False,invert_y=False,plot_all_data=True,s=(16,9)):
    """
    Grafica la data segun dos features a eleccion y separada por clusters

    Parametros:
    df= Toda la data sin ningun tipo de procesamiento.
    data= Lista de DataFrames, donde cada DataFrame contiene los datos de cada cluster.
    feature_x= Feature del eje x
    feature_y= Feature del eje y

    """

    fig = plt.figure(figsize=s)
    ax= fig.add_subplot(111)
    if plot_all_data == True:
        xdf,ydf= apply_log10(df[feature_x],df[feature_y],semilogx,semilogy)
        ax.scatter(xdf,ydf,c="gray",label="All data",alpha=0.15)
    for i in range(len(data)):
        xdata,ydata= apply_log10(data[i][feature_x],data[i][feature_y],semilogx,semilogy)
        l= "cluster "+str(i+1)
        ax.scatter(xdata,ydata,c="C"+str(i),label=l)

    ax.legend()
    if semilogx == True:
        ax.set_xlabel("Log("+feature_x+")",fontsize=15)
    else:
        ax.set_xlabel(feature_x,fontsize=15)
    if semilogy == True:
        ax.set_ylabel("Log("+feature_y+")",fontsize=15)
    elif feature_y == "logLbol":
        ax.set_ylabel("Log(L/Lsun)",fontsize=15)
    else:
        ax.set_ylabel(feature_y,fontsize=15)
    if invert_x==True:
        ax.invert_xaxis()
    if invert_y==True:
        ax.invert_yaxis()
    ax.grid()
