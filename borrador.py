def get_clusters2(data,labels,r="r2"):

    """
    Dos listas para la data transformada con PCA u otro metodo y para la
    data sin normalizar. Cada elemenro de estas listas representa a un cluster distinto.
    """

    clusters_index, noise_index= get_cluster_index(labels)
    data = data.reset_index(drop=True)
    #xstd = xstd.reset_index(drop=True)
    clusters_data=[]
    for i in range(len(clusters_index)):
        clusters_data.append(data.drop(data.index[clusters_index[i]]))
        #clusters_xstd.append(xstd.drop(xstd.index[clusters_index[i]]))
    if noise_index:
        noisy_data= data.drop(data.index[noise_index])
    else:
        noisy_data= []
        #noisy_data_xstd= pd.DataFrame([])
    if r=="r1":
        return clusters_data, noisy_data
    if r== "r2":
        return clusters_data


def get_spectral_class(data):
    sclass= []
    for i in range(data.shape[0]):

        teff= data.iloc[i]["Teff"]

        if teff>= 30000:
            sclass.append("O")

        elif teff >= 10000 and teff < 30000:
            sclass.append("B")

        elif teff >= 7500 and teff < 10000:
            sclass.append("A")

        elif teff >= 6000 and teff < 7500:
            sclass.append("F")

        elif teff >= 5200 and teff < 6000:
            sclass.append("G")

        elif teff >= 3700 and teff < 5200:
            sclass.append("K")

        elif teff >= 2400 and teff < 3700:
            sclass.append("G")
        else:
             sclass.append("Undetermined")
    return sclass

def get_spectral_class(data):
    sclass= []
    for i in range(data.shape[0]):
        teff= data.iloc[i]["Teff"]
        mass= data.iloc[i]["M1"]
        radius= data.iloc[i]["Rad2"]
        lum= data.iloc[i]["Lbol"]

        if teff>= 30000:
            if mass >= 16 and radius >= 6.6 and lum >=30000:
                sclass.append("O")

        elif teff >= 10000 and teff < 30000:
            if mass >= 2.1 and mass <16:
                if radius >= 1.8 and radius < 6.6:
                    if lum >= 25 and lum < 30000:
                         sclass.append("B")

        elif teff >= 7500 and teff < 10000:
            if mass >= 1.4 and mass <2.1:
                if radius >= 1.4 and radius < 1.8:
                    if lum >= 5 and lum < 25:
                         sclass.append("A")

        elif teff >= 6000 and teff < 7500:
            if mass >= 1.04 and mass <1.4:
                if radius >= 1.15 and radius < 1.4:
                    if lum >= 1.5 and lum < 5:
                        sclass.append("F")

        elif teff >= 5200 and teff < 6000:
            if mass >= 0.8 and mass <1.04:
                if radius >= 0.96 and radius < 1.15:
                    if lum >= 0.6 and lum < 1.5:
                         sclass.append("G")

        elif teff >= 3700 and teff < 5200:
            if mass >= 0.45  and mass < 0.8:
                if radius >= 0.7 and radius < 0.96:
                    if lum >= 0.08 and lum < 0.6:
                        sclass.append("K")

        elif teff >= 2400 and teff < 3700:
            if mass >= 0.08 and mass < 0.45:
                if radius < 0.7:
                    if lum < 0.08:
                        sclass.append("G")
        else:
             sclass.append("Undetermined")
    return sclass


def plot_spectral_hist(data):
    m= int(np.ceil(len(data)/2))
    fig= plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(m, 2)

    for i in range(m):
        if 2*i <= len(data)-1:
            ax= fig.add_subplot(gs[i, 0])
            ax= data[2*i]["Sclass"].hist()
            ax.set_title("Cluster"+str(2*i+1))
        if 2*i+1 <= len(data)-1:
            ax= fig.add_subplot(gs[i, 1])
            ax= data[2*i+1]["Sclass"].hist()
            ax.set_title("Cluster"+str(2*i+2))
    plt.tight_layout()


plot_spectral_hist(all_data)
