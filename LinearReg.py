import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#Libraries for models
from sklearn.linear_model import LinearRegression

#Liblaries for cross validation and model evaluation
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# In[26]:
def load_dataset(path):
    dataset=pd.read_csv(path)
    return dataset

dataset = load_dataset("dataset/vehicles.csv")

# In[ ]:
#Se eliminan valores de atributos poco relevantes
dataset.drop(['id', 'url', 'region', 'region_url', 'image_url', 'title_status', 'size', 'VIN', 'description', 'county', 'lat', 'long', 'posting_date'], axis=1, inplace=True)

# In[ ]:
'''Tratamiento de datos nan'''
    #price
dataset['price'].replace(0,np.nan,inplace=True)
dataset['price'].replace(1,np.nan,inplace=True)
dataset.dropna(subset=['price'], inplace = True)

    #condition
#cont_condition = dataset['condition'].value_counts()
#print(cont_condition)
dataset['condition'].replace(np.nan, 'fair', inplace=True)

    #year
mean_year = dataset['year'].astype(float).mean(axis=0)
dataset['year'].replace(np.nan,mean_year, inplace = True)

    #fuel
dataset['fuel'].replace(np.nan,'gas',inplace = True)

    #odometer
mean_odometer = dataset['odometer'].astype(float).mean(axis=0)
dataset['odometer'].replace(np.nan,mean_odometer, inplace = True)

    #cylinders
dataset['cylinders'].replace(np.nan,'6 cylinders', inplace = True)

    #transmission
dataset['transmission'].replace(np.nan,'automatic', inplace = True)

    #drive
dataset['drive'].replace(np.nan,'4wd', inplace = True)

    #type
dataset['type'].replace(np.nan,'other',inplace = True)

# In[ ]:
'''Tratamiento de datos avanzado'''
#modificamos el valor year para que tenga los años 
#del vehiculo y no el año de fabricación
dataset['year'] = (2022 - dataset['year']).astype(int)
#dataset= dataset[dataset['year'] < 50]
# Identificamos el numero de coches para cada fabricante y nos centramos
#en las 10 con mas coches, el resto se agregan a un nuevo subgrupo 
mf = dataset['manufacturer'].value_counts()
dataset['manufacturer'] = dataset['manufacturer'].apply(lambda x: x if str(x) in mf[:10] else 'others')
# establecemos un rango de precio donde se centran la malloria de los datos
#asi quitamos posibles errores e irrelevantes
dataset = dataset[dataset['price'] > 1000]
dataset = dataset[dataset['price'] < 60000]
kms_up = dataset['odometer'].quantile(0.99) # Upper
kms_down = dataset['odometer'].quantile(0.1)  # Lower
# en el dataset se quedaran los datos que esten dentro de este rango
# obviando valores irrelevantes o posibles errores
dataset = dataset[(dataset['odometer'] < kms_up) & (dataset['odometer'] > kms_down)]


'''matriz de correlacion'''
sns.heatmap(dataset.corr(), annot=True)
plt.savefig('Graficas/matriz_correlacion/' + ' ' + 'corr_matrix' + '.png')

dataset["cylinders"].replace({"6 cylinders":"6","8 cylinders":"8","4 cylinders":"4",
                              "5 cylinders":"5", "3 cylinders":"3","10 cylinders":"10",
                              "12 cylinders":"12","other":"6"}, inplace = True)
dataset[["cylinders"]] = dataset[["cylinders"]].astype("int")

# In[ ]:
'''Regresion lineal 1'''
X = dataset[['year', 'odometer']]
y = dataset[['price']] 

#primera regresion lineal observando edad, kms y precio
lm = LinearRegression()
lm.fit(X,y)
R2_lm = lm.score(X,y)
#prediccion del precio por cada vehiculo observando la edad y los kms
lm_prediccion_precio = lm.predict(X)
mae_lm = mean_absolute_error(dataset['price'], lm_prediccion_precio)
mse_lm = mean_squared_error(dataset['price'], lm_prediccion_precio)

# In[]:
'''transformacion de datos categoricos a numericos'''
    #condition
condiciones= pd.get_dummies(dataset['condition']).reset_index()
    #transmision
transmisiones = pd.get_dummies(dataset['transmission']).reset_index()
transmisiones.rename(columns = {'other':'other_transmission'}, inplace = True)
    #tipo de vehiculo
tipos_vehiculo = pd.get_dummies(dataset['type']).reset_index()
tipos_vehiculo.rename(columns = {'other':'other_type'}, inplace =True)
    #tracciones
tracciones = pd.get_dummies(dataset['drive']).reset_index()
    #cilindrada
cilindrada = pd.get_dummies(dataset['cylinders']).reset_index()
    #combustible
combustible = pd.get_dummies(dataset['fuel']).reset_index()

dum_ds = pd.merge(condiciones, transmisiones, on = 'index')
dum_ds = pd.merge(dum_ds, combustible, how = 'inner')
dum_ds = pd.merge(dum_ds, tipos_vehiculo, how = 'inner')
dum_ds = pd.merge(dum_ds, tracciones, how = 'inner')
dum_ds = pd.merge(dum_ds, cilindrada, how = 'inner')

# In[]:
'''Regresion lineal 2'''
X2 = dataset[['year', 'odometer']].reset_index()
X2 = pd.merge(X2, dum_ds, on = 'index')
X2 = X2.drop(columns = ['index'])

y2 = dataset[['price']]

lm2 = LinearRegression()
lm2.fit(X2,y2)
lm2_prediccion_precio= lm2.predict(X2)
mae_lm2 = mean_absolute_error(dataset['price'], lm2_prediccion_precio)
mse_lm2 = mean_squared_error(dataset['price'], lm2_prediccion_precio)
R2_lm2 = r2_score(dataset['price'], lm2_prediccion_precio)
# In[]:
figure = plt.figure(figsize=(25, 7))
sns.histplot(data=lm2_prediccion_precio, color = 'b', kde=True)
sns.histplot(data=dataset['price'], color = 'r', kde=True)
plt.xlim(0,60000)
plt.savefig('Graficas/comparacion_prediccion/' + ' ' + 'linear_reg_pred' + '.png')

# In[]:
'''Regresion lineal 3'''
x_train, x_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.3, random_state = 4222)
lm3 = LinearRegression()
lm3.fit(x_train, y_train)
#cross validation con k = 5
cross = cross_val_score(lm3,X2,y2)
cross_prediccion_precio = cross_val_predict(lm3,X2,y2)
R2 = r2_score(dataset['price'], cross_prediccion_precio)
mae_cv = mean_absolute_error(dataset['price'], cross_prediccion_precio)
mse_cv = mean_squared_error(dataset['price'], cross_prediccion_precio)
# In[]:
figure = plt.figure(figsize=(25, 7))
sns.histplot(data=cross_prediccion_precio, color = "b",kde=True)
sns.histplot(data=dataset['price'], color = "r",kde=True)
plt.xlim(0,60000)
plt.savefig('Graficas/comparacion_prediccion/' + ' ' + 'cross_val_pred' + '.png')