import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[26]:


def load_dataset(path):
    dataset = pd.read_csv(path)
    return dataset


dataset = load_dataset("dataset/vehicles.csv")

# In[ ]:

# Se eliminan valores de atributos poco relevantes
dataset.drop(['id', 'url', 'region', 'region_url', 'image_url', 'title_status', 'size',
             'VIN', 'description', 'county', 'lat', 'long', 'posting_date'], axis=1, inplace=True)

# In[ ]:
'''Tratamiento de datos nan'''
# price
dataset['price'].replace(0, np.nan, inplace=True)
dataset['price'].replace(1, np.nan, inplace=True)
dataset.dropna(subset=['price'], inplace=True)

# condition
#cont_condition = dataset['condition'].value_counts()
# print(cont_condition)
dataset['condition'].replace(np.nan, 'fair', inplace=True)

# year
mean_year = dataset['year'].astype(float).mean(axis=0)
dataset['year'].replace(np.nan, mean_year, inplace=True)

# fuel
dataset['fuel'].replace(np.nan, 'gas', inplace=True)

# odometer
mean_odometer = dataset['odometer'].astype(float).mean(axis=0)
dataset['odometer'].replace(np.nan, mean_odometer, inplace=True)

# cylinders
dataset['cylinders'].replace(np.nan, '6 cylinders', inplace=True)

# transmission
dataset['transmission'].replace(np.nan, 'automatic', inplace=True)

# drive
dataset['drive'].replace(np.nan, '4wd', inplace=True)

# type
dataset['type'].replace(np.nan, 'other', inplace=True)

# In[ ]:
'''Tratamiento de datos avanzado'''
# modificamos el valor year para que tenga los años
# del vehiculo y no el año de fabricación
dataset['year'] = (2022 - dataset['year']).astype(int)

# Identificamos el numero de coches para cada fabricante y nos centramos
# en las 10 con mas coches, el resto se agregan a un nuevo subgrupo
mf = dataset['manufacturer'].value_counts()
dataset['manufacturer'] = dataset['manufacturer'].apply(
    lambda x: x if str(x) in mf[:10] else 'others')
# establecemos un rango de precio donde se centran la malloria de los datos
# asi quitamos posibles errores e irrelevantes
dataset = dataset[dataset['price'] > 1000]
dataset = dataset[dataset['price'] < 60000]
kms_up = dataset['odometer'].quantile(0.99) # Upper
kms_down = dataset['odometer'].quantile(0.1)  # Lower
# en el dataset se quedaran los datos que esten dentro de este rango
# obviando valores irrelevantes o posibles errores
dataset = dataset[(dataset['odometer'] < kms_up) & (dataset['odometer'] > kms_down)]



dataset["cylinders"].replace({"6 cylinders": "6", "8 cylinders": "8", "4 cylinders": "4",
                              "5 cylinders": "5", "3 cylinders": "3", "10 cylinders": "10",
                              "12 cylinders": "12", "other": "6"}, inplace=True)
dataset[["cylinders"]] = dataset[["cylinders"]].astype("int")
# In[]:
'''Histogramas Atributos'''
# year
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['year'])
plt.xlim(0, 40)
plt.savefig('Graficas/histogramas/' + ' ' + 'year' + '.png')
# kms
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['odometer'])
plt.xlim(15000, len(dataset['odometer'].unique()))
plt.savefig('Graficas/histogramas/' + ' ' + 'odometer' + '.png')
# price
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['price'])
plt.xlim(0, 60000)
plt.savefig('Graficas/histogramas/' + ' ' + 'price' + '.png')
# manufacturer
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['manufacturer'])
plt.xlim(0, len(dataset['manufacturer'].unique()))
plt.xticks(rotation=90)
plt.ylim(0, 120000)
plt.savefig('Graficas/histogramas/' + ' ' + 'manufacturer' + '.png')
# condition
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['condition'])
plt.xlim(0, len(dataset['condition'].unique()))
plt.ylim(0, 200000)
plt.savefig('Graficas/histogramas/' + ' ' + 'condition' + '.png')
# type
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['type'])
plt.xlim(0, len(dataset['type'].unique()))
plt.xticks(rotation=90)
plt.ylim(0, 100000)
plt.savefig('Graficas/histogramas/' + ' ' + 'type' + '.png')
# cylinder
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['cylinders'])
plt.xlim(0, len(dataset['cylinders'].unique()))
plt.savefig('Graficas/histogramas/' + ' ' + 'cylinders' + '.png')
# fuel
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['fuel'])
plt.xlim(0, len(dataset['fuel'].unique()))
plt.savefig('Graficas/histogramas/' + ' ' + 'fuel' + '.png')
# drive
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['drive'],)
plt.xlim(0, len(dataset['drive'].unique()))
plt.ylim(0, 200000)
plt.savefig('Graficas/histogramas/' + ' ' + 'drive' + '.png')
# paint_color
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['paint_color'])
plt.xlim(0, len(dataset['paint_color'].unique()))
plt.xticks(rotation=90)
plt.ylim(0, 60000)
plt.savefig('Graficas/histogramas/' + ' ' + 'paint_color' + '.png')
# transmission
figure = plt.figure(figsize=(7, 4))
sns.histplot(data=dataset['transmission'])
plt.xlim(0, len(dataset['transmission'].unique()))
plt.savefig('Graficas/histogramas/' + ' ' + 'transmission' + '.png')
# state
figure = plt.figure(figsize=(12, 4))
sns.histplot(data=dataset['state'])
plt.xlim(0, len(dataset['state'].unique()))
plt.xticks(rotation=90)
plt.savefig('Graficas/histogramas/' + ' ' + 'state' + '.png')

# In[ ]:
'''diagramas de cajas'''
# precio vs condicion
fig = plt.figure(figsize=(15, 8))
sns.boxplot(data=dataset, x='condition', y='price')
plt.xticks(rotation=90)
plt.savefig('Graficas/cajas/' + ' ' + 'price_condition' + '.png')
# precio vs fabricante
fig = plt.figure(figsize=(15, 8))
sns.boxplot(data=dataset, x='manufacturer', y='price')
plt.xticks(rotation=90)
plt.savefig('Graficas/cajas/' + ' ' + 'price_manufacturer' + '.png')
# precio vs tipo
fig = plt.figure(figsize=(15, 8))
sns.boxplot(data=dataset, x='type', y='price')
plt.xticks(rotation=90)
plt.savefig('Graficas/cajas/' + ' ' + 'price_type' + '.png')
# precio vs tipo de transmision
fig = plt.figure(figsize=(15, 8))
sns.boxplot(data=dataset, x='drive', y='price')
plt.xticks(rotation=90)
plt.savefig('Graficas/cajas/' + ' ' + 'price_drive' + '.png')
# precio vs estado
fig = plt.figure(figsize=(15, 8))
sns.boxplot(data=dataset, x='state', y='price')
plt.xticks(rotation=90)
plt.savefig('Graficas/cajas/' + ' ' + 'price_state' + '.png')

# In[]:
#dataset_year = dataset[dataset['year'] < 60]
fig = plt.figure(figsize=(25, 8))
aux=plt.subplot()
aux=sns.scatterplot(data=dataset, x='year', y='price')
#aux=sns.scatterplot(data=dataset_year , x='year', y='price')
plt.xticks(rotation=90)
# In[]:
contador_traccion = dataset['drive'].value_counts().to_frame()
contador_traccion.rename(columns={'drive': 'value_counts'}, inplace=True)

dataset_drive_type = dataset[['drive','type','price']]
dataset_drive_type = dataset_drive_type.groupby(['drive','type'],as_index=False).mean()

price_drive_type = dataset_drive_type.pivot(index='drive',columns='type')

