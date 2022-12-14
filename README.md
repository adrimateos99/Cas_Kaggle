# Pràctica Kaggle APC UAB 2022-23
### Nom: Adrià Mateos Martínez
### Dataset: Dades de vehicles per vendre de segona má
### URL: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

## Resum:
El dataset conté 426880 instàncies amb 26 atributs cadascuna. 6 d'aquests atributs són numèrics, 1 te format de data i la resta són categòrics. 
Els atributs representen característiques generals del vehicle com el fabricant o la marca i també específiques com el quilometratge que té, el combustible que utilitza o el tipus de tracció.

## Objectius del dataset.
L'objectiu principal serà predir el preu de cada vehicle en funció dels valors d'entrada.

## Enteniment de les dades.
Els atributs que més correlació poden tenir amb la sortida (atribut 'price') són el quilometratge i l'edat del vehicle. Així i tot, es pot observar a la matriu de correlació que només n'hi ha amb l'atribut edat del vehicle ('year'). També s'analitza que mentre augmenta l'edat, disminueix el preu. Tot i això, quan un vehicle arriba als 50 anys es veu com el preu d'aquest torna a augmentar ja que és considerat històric i el seu valor augmenta. Es podria dir que durant els primers 50 anys de vida del vehicle la correlació entre edat i preu es negativa i és a partir dels 50 que aquesta es torna positiva. Es podria suposar que en el cas dels atributs quilometratge ('odometer') i preu també podria haver-hi correlació positiva, ja que es pot pensar que mentre augmenten els anys de vida del vehicle, aquest continua circulant. Però no es així, pel fet que els cotxes antics són els que menys circulen i, per tant, no augmenten linearment els quilometres circulats a l'edat que tenen. Pel que fa a la resta de dades, al ser categóriques és necessari un processament d'aquestes i s'explicarà la possible correlació amb l'atribut objectiu més endevant.

## Processament de les dades.
S'han eliminat atributs irrellevants com: 'id', 'url', 'region', 'region_url', 'image_url', 'title_status', 'size', 'VIN', 'description', 'county', 'lat', 'long', 'posting_date'. Aquests atributs eren irrellevants per a la realització de la pràctica i molt d'aquest també tenien moltes dades NaN. Per tractar les dades Nan, s'ha obtat per modificar aquestes dades per la moda de l'atribut.

Durant la realització de la pràctica s'ha observat que hi havia dades que no tenien coherència emb la resta d'atributs del vehicle, és per aixó que s'ha establert un rang de preus (de 1000 a 60000), d'aquesta manera s'han evitat possibles dades errònies i s'han eliminat vehicles irrellevant, ja que podien ser exageradament cars o barats. També s'ha realitzat una modificació en l'atribut 'year' perquè mostri l'edat del vehicle i no l'any de matriculació. Pel cas de l'atribut fabricant, hi havia molts que eren irrellevants, per tant, s'ha obtat per sumar tots els fabricants amb menys de 10000 vehicles i ajuntar-los en un nou tipus de fabricant ('others').

En el cas dels atributs categorics 'condition', 'transmission', 'type', 'drive', 'cylinders', 'fuel', s'ha obtat pasar-los a numèric mitjançant la funció get_dummies.

## Probes realitzades.
S'han realitzat 2 tipus de regresions lineals i l'última s'ha validat mitjançant 'cross_validation' amb k = 5. La primera regressió s'ha fet unicament utilitzant els atributs de l'edat i el quilometratge. La segona s'ha realitzat mitjançant tots els atributs categòrics que s'han passat a numèrics, l'edat i el quilometratge.

###Resultats
|--|R2score|Mean_absolute_error|Mean_square_error|
|--|--|--|--|
| Linear Regression 1 | 0.3384 | 7693.4659 | 99004700.2530 |
| Linear Regression 2 | 0.6108 | 5660.1417 | 58229190.1161 |
| Cross Validation | 0.6081 | 5680.8522 | 58638024.0404 |

Es pot observar que la segona Regressió Lineal, concretament la que disposa de tots el atributs obté millors resultats. També s'observa com la validació confirma els resultats obtinguts. El Regressor Lineal té un 60% d'encerts en el que a predir el preu es tracta.

Per tal d'evitar observar dades de vehicles historics on el preu augmenta segons l'edat, s'ha realitzat la Regresseió Lineal mitjançant dades de vehicles de menys de 50 anys.

### Resultats
|--|R2score|Mean_absolute_error|Mean_square_error|
|--|--|--|--|
| Linear Regression 1 | 0.3725 | 7506.6474 | 93529861.9580 |
| Linear Regression 2 | 0.6487 | 5388.7116 | 52358286.0756 |
| Cross Validation | 0.6462 | 5409.2036 | 52731503.8048 |

S'bserva que el percentatge d'encerts ha augmentat fins al 65%.

## Conclusions.
Ja que el percentatge obtingut pel Regressor Lineal no és extremadament bó, no podem afirmar al 100% que el preu d'un vehicle estigui directament relacionat a les seves característiques. També es interessant mencionar que la base de dades conté diferents tipus de vehicles i, per tant, no tots tenen el mateix valor a futur ni tots devaluen de la mateixa manera. Per exemple, un cotxe històric tindrà més valor que un camió històric.

Certament podriem establir un rang de preus de -5500 a 5500 on el preu predit fos 0. D'aquesta manera podem considerar que el preu real es troba dins del rang de preus. Quant més petit sigui el rang de preus inicial (l'utilitzat pel Regressor Lineal es de 1000 a 60000), més precisió obtindrem alhora de predir el preu.

Pel que fa a la Regressió Lineal on s'utilitzen dades de vehicles amb menys de 50 anys, no podem considerar tampoc per suficientment bons els resultats. El rang de preus passaria de ser d'11000 a 10600 sent una baixada mínima.


### Coments
*El dataset ocupa massa espai com per pujar-lo al github, motiu pel cual no he fet commits ja que tenia que esta eliminant el dataset per fer-lo i era mmolt engorrós. El codi te com a direccio del dataset: dataset/vehicles.csv*
