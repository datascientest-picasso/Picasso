import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from skimage.color import rgb2gray
from skimage import io
from sklearn.utils import shuffle
from skimage.transform import rescale, resize
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical

def preprocess_image(image_path):

    im2 = io.imread(image_path)
    gray = rgb2gray(im2)
    gray = resize(gray, (200, 200), anti_aliasing=True)
    img = gray.astype(float)
    img = np.ravel(img)
    return img

def load_images():

	all_paint=[]
	Path = 'Pinturas/'
	
	artists=os.listdir(Path)
	for artist in artists:
  	  paint=os.listdir(Path+artist)
  	  for paints in paint:
     	   all_paint.append(preprocess_image(Path+artist+'/'+paints))

	tamanho_pinturas = len(all_paint)
	print("Tamanho do arranjo pinturas = ",tamanho_pinturas)
	
	return all_paint
	
def load_artistas():
	artistas=[]
	Path = 'Pinturas/'
	artists=os.listdir(Path)
	for artist in artists:
		paint=os.listdir(Path+artist)
		for paints in paint:
			artistas.append(artist)
     	   
	tamanho_artistas = len(artistas)
	print("Tamanho do arranjo artistas = ",tamanho_artistas)
    
	return artistas

def image_to_pandas(image,artista):
	df = pd.DataFrame(image)
	df.loc[:, 'pintores'] = artista
	temp_df = df
	temp_df.loc[:,'pintores'] = pd.factorize(temp_df.pintores)[0]
	df = shuffle(temp_df)
	return df

def Keras():
	n = 50
	model = Sequential()
	model.add(Dense(n, input_shape=(40000,), activation='relu') )
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(150,activation='softmax'))
	model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.summary() 
	return model
	
def Keras2():
	n = 50
	model = Sequential()
	model.add(Dense(n, input_shape=(40000,), activation='relu') )
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(n,activation='relu'))
	model.add(Dense(49,activation='softmax'))
	model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.summary() 
	return model	

def Keras3():
	model = Sequential()
	model.add(Dense(10, input_shape=(40000,), activation='relu') )
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(49,activation='softmax'))
	model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.summary() 
	return model	

carregar_artistas = 1

if carregar_artistas == 1:
	print("\n")
	print("Carregando pinturas")
	pinturas = load_images()

	print("\n")
	print("Carregando nomes dos pintores")
	pintores = load_artistas()

	print("\n")
	print("Convertendo para Pandas DataFrame")
	df = image_to_pandas(pinturas,pintores)
	print("Formato do DataFrame")
	print("Linhas = {}".format(df.shape[0]))
	print("Colunas = {}".format(df.shape[1]))
	#df.to_csv("Pinturas.csv", index=True)
else: 
	print("Lendo arquivo CSV")
	df = pd.read_csv("Pinturas.csv", index=True)
	print("Arquivo lido com sucesso")
	print("Formato do arquivo")
	print("Linhas = {}".format(df.shape[0]))
	print("Colunas = {}".format(df.shape[1]))


y = df.pintores
X = df.drop(['pintores'],axis = 1).values#.reshape(-1,200,200,1)

y = to_categorical(y,num_classes=50)
y.astype('int32')
y = np.argmax(y,axis=1)

print("\n")
print("Separando em amostras de teste e treino")

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, y, test_size=0.25, random_state=42)

#Y = Y_teste.astype('int32')
#Y_teste = np.argmax(Y_teste,axis=1)
#Y_treino = np.argmax(Y_treino,axis=1)

#Keras
print("\n")
print("Compilando Keras")

model = Keras()

model.fit(X_treino, Y_treino, epochs=150)

y_pred = model.predict(X_teste)
y_pred2 = np.argmax(y_pred,axis=1)

score_keras = model.evaluate(X_teste, Y_teste)
print("\n")
print("Score = ",score_keras)

score2_keras = accuracy_score(Y_teste,y_pred2)
print("\n")
print("Score 2 = ",score2_keras)

valores_keras = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": y_pred2})
valores_keras.to_csv("previsao_keras.csv", index=False)

#Keras2
print("\n")
print("Compilando Keras 2")

model = Keras2()

model.fit(X_treino, Y_treino, epochs=100)

y_pred = model.predict(X_teste)
y_pred2 = np.argmax(y_pred,axis=1)

score_keras2 = model.evaluate(X_teste, Y_teste)
print("\n")
print("Score = ",score_keras2)

score2_keras2 = accuracy_score(Y_teste,y_pred2)
print("\n")
print("Score 2 = ",score2_keras2)

valores_keras2 = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": y_pred2})
valores_keras2.to_csv("previsao_keras2.csv", index=False)

#Keras2
print("\n")
print("Compilando Keras 3")

model = Keras3()

model.fit(X_treino, Y_treino, epochs=100)

y_pred = model.predict(X_teste)
y_pred2 = np.argmax(y_pred,axis=1)

score_keras3 = model.evaluate(X_teste, Y_teste)
print("\n")
print("Score = ",score_keras3)

score2_keras3 = accuracy_score(Y_teste,y_pred2)
print("\n")
print("Score 2 = ",score2_keras3)

valores_keras3 = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": y_pred2})
valores_keras3.to_csv("previsao_keras3.csv", index=False)