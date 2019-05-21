import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, save_img, img_to_array
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from PIL import Image, ImageFilter
import os
from skimage import color
from skimage import io

def preprocess_image(image_path):
    
    #from keras.applications import vgg19
    img = color.rgb2gray(io.imread(image_path))
    img = np.ravel(img)
    #img = img_to_array(img)
    im=img/255
    img = img.astype(int)
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
	
	#all_paint=np.stack(all_paint)
	
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
	df.loc[:, 'Pintores'] = artistas 
	return df

#pinturas = load_images()
#pintores = load_artistas()

#pinturas = image_to_pandas(pinturas,pintores)

#temp_df = pinturas
#temp_df.loc[:,'Pintores'] = pd.factorize(temp_df.Pintores)[0]

#X_treino, X_teste, Y_treino, Y_teste = train_test_split(all_paint, artistas, test_size=0.25, random_state=42)

#regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=2)

#regr.fit(all_paint, artistas)  

#print("Caracteristicas Random Forest Regressor")
#print(regr.feature_importances_)

#ypred = regr.predict(X_teste)

#kfold = KFold(n_splits=num_folds, random_state=seed)
#cv_resultados = cross_val_score(RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100), X_treino, Y_treino, cv=kfold, scoring=RMS)
#msg = "Modelo: %s Média: %f (Sigma:%f)" % (nome, cv_resultados.mean(), cv_resultados.std())
#print("KFold")
#print(msg)
