import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def categorizar(colnas):
	for col in colnas:
		categorizacion(col)

def categorizacion(col_name):
	top_versions = df[col_name].value_counts().nlargest(10).index
	df[f'{col_name}_grouped'] = df[col_name].apply(lambda x: x if x in top_versions else 'Other')
	# Calculamos la tasa de detección por valor de col_name
	tasa_deteccion = df.groupby(f'{col_name}_grouped')['HasDetections'].mean()


	# Mapeamos esa tasa a cada fila
	df[f'{col_name}_TE'] = df[f'{col_name}_grouped'].map(tasa_deteccion)
	bins = pd.qcut(df[f'{col_name}_TE'], q=4, duplicates='drop')
	num_bins = bins.cat.categories.size

	etiquetas = ['Muy bajo', 'Bajo', 'Medio', 'Alto'][:num_bins]

	df[f'{col_name}_riesgo'] = pd.qcut(df[f'{col_name}_TE'], q=4, labels=etiquetas, duplicates='drop')
	df[f'{col_name}_TE'].drop()
	df[col_name].drop()
	df[f'{col_name}_grouped'].drop()

def llenarNulos(cols):
	for col in cols:
		df[col] = df[col].fillna(-1)

