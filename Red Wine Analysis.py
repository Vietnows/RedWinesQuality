#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[3]:


wineDf = pd.read_csv('wineQuality-red.csv')
wineDf


# In[165]:


wineDf.info()


# #### Fixed.acidity 
# Normalmente se faz referencia ao ácido tartarico, uns dos principais ácidos encontrados em uvas e um dos principais ácidos do vinho.
# 
# #### Volatile.acidity 
# A acidez volátil refere-se aos ácidos destilados a vapor presentes no vinho, em grandes quantidades pode levar à um gosto desagradável. O nível médio de ácido acético em um vinho é inferior a 400 mg/L,embora os níveis possam variar de indetectáveis até 3 g/L.
# 
# #### Citric.acid
# Presente nas uvas em baixa quantidade, nos vinhos o ácido cítrico tem pouca ou nenhuma presença. Nos vinhos tintos desaparece devido à ação de bactérias láticas (fermentação malolática). Sensorialmente é fresco, porém em alguns casos pode apresentar um leve final amargo.
# 
# #### Residual.sugar
# A fermentação de um vinho é feita através do contato do açucar com a levedura, após a fermentação resta o açucar residual. Quando contém até 4 gramas de açucar residual por litro, um vinho pode ser considerado seco, a partir de 25 a 80 gramas é considerado doce ou suave.
# 
# #### Chlorides
# Representa a quantidade de sal contidas nos vinhos.
# 
# #### Free.sulfur.dioxide
# É uma forma livre de SO2, um gás dissolvido que impede o crescimento de microbios e a oxidação do vinho. Quantidades excessivas de SO2 podem inibir a fermentação e causar efeitos sensoriais indesejáveis.
# 
# #### Total.sulfur.dioxide
# O dióxido de enxofre total (TSO2) é a porção de dioxido de enxofre livre (SO2) que está livre no vinho mais a porção que está ligada a outros produtos químicos no vinho.
# 
# #### Density
# A densidade do vinho se refere ao corpo do vinho, à sensação de maior ou menor densidade que a bebida apresenta. A densidade do vinho pode variar de acordo com a densidade da água e o teor percentual de álcool e açúcar.
# 
# #### PH
# O pH (potencial Hidrogeniônico) é calculado a partir da concentração de íons de hidrogênio. Indica acidez, neutralidade ou alcalinidade de um produto. A escala varia de 0 a 14 e, quanto menor for o índice de pH, maior é a acidez. Abaixo de 7, o pH é ácido, igual a 7 é neutro, e maior que 7 é alcalino.
# 
# Nos vinhos em geral, o pH varia de 2,8 (acidez forte) até 3,8 (acidez leve). Com pH acima de 3,5 o vinho é frágil e pode estar sujeito a alterações (defeitos). Um pH baixo tem grande importância na estabilidade do vinho.
# 
# #### Sulphates
# O termo sulfato é um termo inclusivo para o dióxido de enxofre (SO2), um conservante que é amplamente utilizado na produção de vinho (e na maioria das indústrias alimentícias) por suas propriedades antioxidantes e antibacterianas. O SO2 desempenha um papel importante na prevenção da oxidação e na manutenção da frescura de um vinho.
# 
# #### Alcohol
# Esta variável se refere a porcentagem de alcool contida nos vinhos.
# 
# O álcool é a alma do vinho. É a sua maior ou menor presença que define muitas das vezes a sua qualidade. É habitual dizer-se de um vinho com mais de 13% de álcool que é encorpado, vinoso, capitoso, quente. Já um vinho seco com menos de 11% de álcool é um vinho leve, magro, ligeiro e quase sempre desinteressante. Mas álcool em excesso pode tornar um vinho pesado, chato, mole, desinteressante.

# A partir do histograma de residual sugar é possivel perceber que tem uma grande distribuição entre 2 à 4, o que posso supor que grande parte dos vinhos analisados são vinhos mais doces.
# 
# O teor alcoolico dos vinhos abaixo de 13% pode indicar vinhos mais leves, maior refrescancia. E o Ph está na média de 3,3. Podendo indicar que os vinhos bem avaliados sejam mais secos e om maior acidez.

# In[4]:


wineDf.describe()


# In[5]:


wineDf['quality'].unique()


# In[6]:


#Função para avaliar agrupar a qualidade em grupos

def evaluateQuality(quality):
    if quality <= 4:
        return 'bad'
    elif quality >=5 and quality <=6:
        return 'moderate'
    else: return 'good'


# In[7]:


wineDf['quality Evaluate'] = wineDf['quality'].apply(evaluateQuality)


# In[8]:


plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.countplot(x='quality Evaluate',data=wineDf)


# A primeira coisa a se perceber no plot acima é que a maioria dos vinhos são de qualidade média. Os valores outliers são bem aproximados a vinhos de qualidades ruins e boas, será que os vinhos usados nas avaliações dos especialistas são de uma região especifica? De várias regiões?

# In[168]:


plt.figure(figsize=(20,6))
plt.title('Distribuição Alcohol', family='Arial', fontsize=15)
wineDf[(wineDf['alcohol']>7) & (wineDf['alcohol']<14)]['alcohol'].hist()


# In[10]:


sns.pairplot(wineDf)


# In[163]:


plt.figure(figsize=(15,10))
sns.heatmap(wineDf.corr(),annot=True)


# Os valores que tem maior correlação com a qualidade são alcohol e volatile acidity.
# 
# Ph tem uma forte correlação negativa com os ácidos (quanto menor o ph maior acidez), porém com volatile acidity há correlação positiva.
# 
# Sulphates e Chlorides tem correlação média, porém somente sulphates tem correlação com a qualidade.
# 
# Density tem forte correlação com os ácidos fixed acidity e citric acid, residual sugar e uma forte correlação negativa com alcohol.
# 
# é possivel perceber que a qualidade dos vinhos tem um relacionamento forte com a qualidade
# Fixed acidity tem o maior relacionento com densidade.
# 

# In[12]:


wineDf.corr()[['quality','alcohol']]*100


# In[17]:


plt.figure(figsize=(12,8))
_ = plt.plot(wineDf['fixed acidity'],wineDf['density'], marker='.', linewidth=0, color='orange')
_ = plt.grid(which='major', color='#cccccc', alpha=0.45)
_ = plt.title('Red Wines - fixed.acidity vs density', family='Arial', fontsize=12)
_ = plt.xlabel('fixed.acidity')
_ = plt.ylabel('density')
_ = plt.show()


plt.figure(figsize=(12,8))
_ = plt.plot(wineDf['alcohol'],wineDf['pH'], marker='.', linewidth=0, color='blue')
_ = plt.grid(which='major', color='#cccccc', alpha=0.45)
_ = plt.title('Red Wines - Alcohol vs pH', family='Arial', fontsize=12)
_ = plt.xlabel('Alcohol')
_ = plt.ylabel('pH')
_ = plt.show()


# In[55]:


wineDf


# In[17]:


wineDf.iloc[:,0:11]


# In[18]:


#CALCULOS ELBOW

def calculate_wcss(data,range1=2,range2=21):
    from sklearn.cluster import KMeans
    wcss = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n, init='k-means++',max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss


def optimal_number_of_clusters(wcss):
    """   
    Parametros
    ----------
    wcss : lista
        lista contendo os valores de soma de quadrados intra-cluster
    Returns
    -------
    int : número de clusters 
    """
    from math import sqrt
    x1, y1 = 2, wcss[0]
    x2, y2 = 21, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]

        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    return distances.index(max(distances)) + 2

#PLOT ELBOW
def elbowPlot(range1,range2,n_cluster, w, title='Método Elbow'):
    #plt.figure(figsize=(12,6))
    plt.figure(figsize=(15,10))
    plt.plot(range(range1,range2), w, linewidth = 2, marker='D',markersize=5)
    plt.title(title, fontsize=14)
    plt.xlabel('Numero de Clusters', fontsize=12)
    plt.ylabel('WCSS (inertia)',  fontsize=12)
    plt.grid(which='both',color='black', axis='x', alpha=0.5)

    plt.axvline(x=n_cluster,linewidth=2,color='red',linestyle='--')


# In[29]:


sumSquares = calculate_wcss(wineDf.iloc[:,0:12])
#sum_of_squares = wcss
n = optimal_number_of_clusters(sumSquares)

print('N° Clusters = {}'.format(n))
elbowPlot(2,21,n,sumSquares,title='Método Elbow - Red Wines')


# ### Elbow para Wines

# A ideia do Elbow é rodar o KMeans para vários quantidades diferentes de clusters e dizer qual dessas quantidades é o número ótimo de clusters. 
# 
# O que geralmente acontece ao aumentar a quantidade de clusters no KMeans é que as diferenças entre clusters se tornam muito pequenas, e as diferenças das observações intra-clusters vão aumentando.
# 
# -Calcular a função de custo, a soma dos quadrados das distâncias internas dos clusters, e traçá-la em um gráfico. 

# In[19]:


wines = wineDf.iloc[:,0:12] 


# In[20]:


from pylab import rcParams
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 14,8
sns.set_style('whitegrid')


# #### Normalização Standard Scaler

# In[21]:


X_data = wines.values.astype('float32', copy=False)
scaler = StandardScaler().fit(X_data)

winesScaler = scaler.transform(X_data)


# In[56]:


winesScaler


# In[26]:


sumSquares_winesScaler = calculate_wcss(winesScaler)
#sum_of_squares = wcss
n_sumSquares_winesScaler = optimal_number_of_clusters(sumSquares_winesScaler)

print('N° Clusters = {}'.format(n_sumSquares_winesScaler))
elbowPlot(2,21,n_sumSquares_winesScaler,sumSquares_winesScaler ,title='Método Elbow - Wines Standard Scaler')


# #### Normalização Min Max Scaler

# In[23]:


#X_data = X_Outlet.values.astype('float32', copy=False)
mms = MinMaxScaler().fit(X_data)

winesMmsScaler = scaler.transform(X_data)


# In[25]:


sumSquares_winesMMSScaler = calculate_wcss(winesMmsScaler)
#sum_of_squares = wcss
n_winesMssScaler = optimal_number_of_clusters(sumSquares_winesMMSScaler)

print('N° Clusters = {}'.format(n_winesMssScaler))
elbowPlot(2,21,n_winesMssScaler,sumSquares_winesMMSScaler ,title='Método Elbow - Wines MinMax Scaler')


# In[27]:


kmeans = KMeans(n_clusters=7)
clustersWines = kmeans.fit_predict(winesMmsScaler)
labelsWines = kmeans.labels_

x1, x2 = 2, 20
intervalo = range(x1,x2+1)


# In[28]:


def plot_clustering(data, labels, title=None):
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(1, figsize=(4, 3))
    plt.figure(figsize=(6, 4))
    plt.scatter(data[:, 0], data[:, 1],
                 c=labels.astype(np.float))
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 


# In[32]:


plot_clustering(winesMmsScaler, labelsWines)
plt.figure(figsize=(14,10))
plt.show()


# ### KMEANS

# In[59]:


kmeans.labels_


# In[61]:


kmeans.cluster_centers_


# In[63]:


wineDf['Clusters'] = kmeans.labels_
wineDf


# In[66]:


wineDf['Clusters'].value_counts()


# In[86]:


wineDf[wineDf['Clusters']==6]


# In[90]:


pd.crosstab(wineDf.quality,kmeans.labels_)


# In[72]:


sns.scatterplot(x='pH',y='density', hue='Clusters', data=wineDf)


# In[74]:


from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# # setting distance_threshold=0 ensures we compute the full tree.
# model = AgglomerativeClustering(distance_threshold=6, n_clusters=6)
# 
# model = model.fit(wines)
# plt.title('Hierarchical Clustering Dendrogram')
# # plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode='level', p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()

# In[81]:


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(wines, 'single')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


# In[162]:


sns.boxplot( y=wines["density"], x=wines["pH"])
plt.show()


# ### Teste Silhouette 
# A análise por Silhouette mede o quão bem um ponto se encaixa em um cluster. Neste método um gráfico é feito medindo quão perto os pontos de um cluster estão dos pontos de outro cluster mais próximo. 
# 
# O coeficiente de Silhouette quando próximo de +1, indica que os pontos estão muito longe dos pontos do outro cluster, e quando próximo de 0, indica que os pontos então muito perto ou até interseccionando um outro cluster.

# In[35]:


from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np


# In[130]:


silhouette_score(winesMmsScaler, labelsWines)


# #### Teste kmeans para 20 clusters calculando o silhoute

# In[131]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=5).fit(winesMmsScaler) for k in range(2,21)]


# In[144]:


silhouete_scores = [silhouette_score(winesMmsScaler, model.labels_) for model in kmeans_per_k[1:]]


# In[146]:


silhouete_scores


# In[157]:


float(np.argmax(silhouete_scores))


# In[156]:


#plt.figure(figsize=(16,5))
plt.plot(range(2,20), silhouete_scores,'bo-', color='blue', linewidth = 3, markersize=8, label='Curva Silhouta')
plt.xlabel('$k$', fontsize='14')
plt.ylabel('Silhoute Score', fontsize='14')
plt.grid(which='major', color='#cccccc', linestyle='--')
plt.title('Número de ótimo de clusters preditos pela curva Silhoute', fontsize=14)
         
#Calcula o numero otimo de clusters
k = np.argmax(silhouete_scores)
plt.axvline(x=k, linestyle='--', c='green', linewidth=3, label='Número ótimo de clusters ({})'.format(k))
plt.scatter(k, silhouete_scores[k-2],c='red',s=400)
plt.legend(shadow=True)         


# #### Este caso o silhoute não perfomou bem, porem iremos rodar o kmeans para varias quantidades de cluster novamente e ver a distribuição dos dados.

# ### Teste kmeans para 10 clusters calculando o silhoute

# In[62]:


range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10]
import matplotlib.cm as cm
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(wines) + (n_clusters + 1) * 10])
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(winesMmsScaler)
# The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(winesMmsScaler, cluster_labels)
    print("Para n_clusters =", n_clusters,
          "O score_silhouette médio é :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(winesMmsScaler, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color,       alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(winesMmsScaler[:, 0], winesMmsScaler[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
plt.show()


# In[164]:


wines


# Quantidade interessante de cluster é de 6 a 7 cluster para esta dataset

# In[ ]:





# #### Regressão Linear

# In[42]:


#X = wineDf.drop(['quality','qualite evaluate'],axis=1)
X = wineDf.iloc[:,0:11]
y = wineDf['quality']


# In[43]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=22)


# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


# Fit the model
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[46]:


lm.coef_


# In[47]:


lm.intercept_


# Fazendo predição 

# In[49]:


prediction = lm.predict(X_test)


# In[51]:


sns.distplot((y_test-prediction),bins=30)
plt.title('Treino vs Predicao')


# ### Avaliando a perfomance do modelo
# 
# Mean Absolute Error (MAE) : é a média do valor absoluto dos erros.
# 
# Mean Squared Error (MSE) : é a média do erro dos quadrados
# 
# Root Mean Squared Error (RMSE) Raiz quadrada do erro dos quadrados

# In[54]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('R2:', lm.score(X_train,y_train))


# In[ ]:




