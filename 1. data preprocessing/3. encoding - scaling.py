
#veri önişleme
import pandas as pd

veriler = pd.read_csv("./datasets/eksikveriler.csv")

#eksik veriler
from sklearn.preprocessing import Imputer #eksik veriler için kullanılır

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) # axis 0 -> column 1 -> row

yas = veriler.iloc[:, 1:4].values #iloc -> kesme | [rows, columns] | a:b -> a ve b arasındakileri(b dahil değil)
imputer = imputer.fit(yas)
yas = imputer.transform(yas)

ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder #her nominal için bir sayısal değer atar
#nominal -> ordinal
labelEncoder = LabelEncoder()
ulke[:,0] = labelEncoder.fit_transform(ulke[:,0])

from sklearn.preprocessing import OneHotEncoder #her sayısal değeri bir kolon haline getirir
#bir hücrenin değerinin olduğu kolon 1 diğerleri 0
#ordinal -> numerik
oneHotEncoder = OneHotEncoder(categorical_features='all')
ulke = oneHotEncoder.fit_transform(ulke).toarray()

yasDF = pd.DataFrame(data = yas, index = range(22), columns = ['boy', 'kilo', 'yas'])
ulkeDF = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr', 'us'])

cinsiyet =  veriler.iloc[:,-1].values
y = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])

x = pd.concat([ulkeDF, yasDF], axis = 1)

from sklearn.cross_validation import train_test_split #dataseti test ve train olarak ikiye böler
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import StandardScaler 
'''normalizasyon veri değerlerini 0 ve 1 arasına ölçeklendirir (MinMaxScaler)
standardizasyon verilerin ortalamasını 0 yapar ve diğer değerler onun alt ve üstünde yer alır
veriler arasında marjinal bir değer olursa normalizasyonda o değer 1 diğerleri 0 olur
normalizasyon için from sklearn.preprocessing import Normalizer'''
sc = StandardScaler() 
X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test) 

