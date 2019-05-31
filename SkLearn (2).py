import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os
from skimage.color import rgb2gray
from skimage import io
from sklearn.utils import shuffle
from skimage.transform import rescale, resize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

def enconder_pint(x,inverse):
	encoder =  LabelEncoder()
	if inverse == 0:
		y = encoder.fit_transform(x)
	if inverse == 1:
		y = encoder.inverse_transform(x)
	return y

def image_to_pandas(image,artista):
	df = pd.DataFrame(image)
	df.loc[:, 'pintores'] = artista
	temp_df = df
	y1 = enconder_pint(temp_df.pintores,0)
	temp_df.loc[:,'pintores'] = y1
	df = shuffle(temp_df)
	return df
	
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
	df = pd.read_csv("Pinturas.csv")
	print("Arquivo lido com sucesso")
	print("Formato do arquivo")
	print("Linhas = {}".format(df.shape[0]))
	print("Colunas = {}".format(df.shape[1]))

y = df.pintores
X = df.drop(['pintores'],axis = 1).values#.reshape(-1,200,200,1)

print("\n")
print("Separando em amostras de teste e treino")

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, y, test_size=0.25, random_state=42)

print("\n")
print("Random Forest Classifier")

#Random Forest Classifier
rfc=RandomForestClassifier(random_state=42,max_features='log2',criterion='gini',max_depth=10,n_estimators=205)

determinar_rfc = 0

if determinar_rfc == 1:
	print("\n")
	print("Determinando melhores parâmetros")
	
	param_grid = { 
		'max_features' : ['auto','sqrt','log2'],
		'n_estimators' : list(range(190,215)),
    	'max_depth' : list(range(8,13))
	}

	CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
	CV_rfc.fit(X_treino, Y_treino)

	print("Exibindo melhores parâmetros")
	melhor_rfc = CV_rfc.best_params_
	print(melhor_rfc)
	
rfc.fit(X_treino, Y_treino)
y_pred_RFC = rfc.predict(X_teste)

valores_rfc = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": y_pred_RFC})
valores_rfc.to_csv("previsao_RFC.csv", index=False)

mse_RFC = np.sqrt(mean_squared_error(Y_teste,y_pred_RFC))
mae_RFC = mean_absolute_error(Y_teste,y_pred_RFC)
accuracy_RFC = accuracy_score(Y_teste,y_pred_RFC)

print("\n")
print("MSE = {}".format(mse_RFC))
print("MAE = {}".format(mae_RFC))
print("Accuracy = {}".format(accuracy_RFC))

cf_RFC = confusion_matrix(Y_teste,y_pred_RFC)

print("\n")
print("Matriz de confusão")
print(cf_RFC)
df_cf_RFC = pd.DataFrame(cf_RFC)
df_cf_RFC.to_csv('MatrizConfusaoRFC.csv')

#Linear Discriminant Analysis

print("\n")
print("Linear Discriminant Analysis")

lda = 	LinearDiscriminantAnalysis()
lda.fit(X_treino, Y_treino)
y_pred_LDA = lda.predict(X_teste)

valores_lda = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": y_pred_LDA})
valores_lda.to_csv("previsao_LDA.csv", index=False)

mse_LDA = np.sqrt(mean_squared_error(Y_teste,y_pred_LDA))
mae_LDA = mean_absolute_error(Y_teste,y_pred_LDA)
accuracy_LDA = accuracy_score(Y_teste,y_pred_LDA)

print("\n")
print("MSE = {}".format(mse_LDA))
print("MAE = {}".format(mae_LDA))
print("Accuracy = {}".format(accuracy_LDA))

cf_LDA = confusion_matrix(Y_teste,y_pred_LDA)

print("\n")
print("Matriz de confusão")
print(cf_LDA)
df_cf_LDA = pd.DataFrame(cf_LDA)
df_cf_LDA.to_csv('MatrizConfusaoLDA.csv')

print("\n")
print("Quadratic Discriminant Analysis")

qda = 	QuadraticDiscriminantAnalysis()
qda.fit(X_treino, Y_treino)
y_pred_QDA = qda.predict(X_teste)

valores_qda = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": y_pred_QDA})
valores_qda.to_csv("previsao_QDA.csv", index=False)

mse_QDA = np.sqrt(mean_squared_error(Y_teste,y_pred_QDA))
mae_QDA = mean_absolute_error(Y_teste,y_pred_QDA)
accuracy_QDA = accuracy_score(Y_teste,y_pred_QDA)

print("\n")
print("MSE = {}".format(mse_QDA))
print("MAE = {}".format(mae_QDA))
print("Accuracy = {}".format(accuracy_QDA))

cf_QDA = confusion_matrix(Y_teste,y_pred_QDA)

print("\n")
print("Matriz de confusão")
print(cf_QDA)
df_cf_QDA = pd.DataFrame(cf_QDA)
df_cf_QDA.to_csv('MatrizConfusaoQDA.csv')

#Decision Tree Classifier
print("\n")
print("Decision Tree Classifier")

dtc = DecisionTreeClassifier()

determinar_dtc = 0

if determinar_dtc == 1:
	print("\n")
	print("Determinando melhores parâmetros")

	param_grid = { 
		'max_depth': list(range(1 ,300)),
		'max_features': list(range(1 ,100))
	}

	CV_dtc = GridSearchCV(estimator=dtc, param_grid=param_grid, cv= 5)
	CV_dtc.fit(X_treino, Y_treino)

	print("Exibindo melhores parâmetros")
	print(CV_dtc.best_params_)
	
dtc = DecisionTreeClassifier(max_depth=2,max_features=57)
dtc.fit(X_treino, Y_treino)
y_pred_DTC = dtc.predict(X_teste)

valores_dtc = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": y_pred_DTC})
valores_dtc.to_csv("previsao_DTC.csv", index=False)

mse_DTC = np.sqrt(mean_squared_error(Y_teste,y_pred_DTC))
mae_DTC = mean_absolute_error(Y_teste,y_pred_DTC)
accuracy_DTC = accuracy_score(Y_teste,y_pred_DTC)

print("\n")
print("MSE = {}".format(mse_DTC))
print("MAE = {}".format(mae_DTC))
print("Accuracy = {}".format(accuracy_DTC))

cf_DTC = confusion_matrix(Y_teste,y_pred_DTC)

print("\n")
print("Matriz de confusão")
print(cf_DTC)
df_cf_DTC = pd.DataFrame(cf_DTC)
df_cf_DTC.to_csv('MatrizConfusaoDTC.csv')