import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from dotenv import load_dotenv
import os
import zipfile

from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ----------------- Función principal para obtener el DataFrame procesado -----------------


# Función para descargar el dataset de Kaggle y descomprimirlo en directorio de trabajo
def download_dataset():
    load_dotenv()
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

    from kaggle.api.kaggle_api_extended import KaggleApi

    print("Autenticando en Kaggle...")
    # Inicializar y autenticar la API
    api = KaggleApi()
    api.authenticate()

    print("Descargando el dataset de la competencia 'microsoft-malware-prediction'...")
    # Descargar los archivos de la competencia
    api.competition_download_files(
        competition="microsoft-malware-prediction",
        quiet=False
    )

    print("Descomprimiendo 'train.csv' del archivo ZIP descargado...")
    # Descomprimir el ZIP descargado
    with zipfile.ZipFile("microsoft-malware-prediction.zip", "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith("train.csv"):
                zip_ref.extract(member)
                break

    # Borrar el ZIP para liberar espacio
    print("Borrando el archivo ZIP descargado para ahorrar almacenamiento...")
    os.remove("microsoft-malware-prediction.zip")
    print("Dataset descargado y descomprimido correctamente.")



# Función para leer el archivo CSV y devolver un DataFrame de pandas
# Esta función permite especificar tipos de datos personalizados y columnas a leer.
def read_malware_csv(file_path = 'train.csv', dtypes_dict=None, columns=None):
    """
    Lee un archivo CSV y devuelve un DataFrame de pandas.

    Args:
        file_path: Ruta al archivo CSV.

    Returns:
        Un DataFrame de pandas con los datos del archivo CSV.
    """

    print(f"Leyendo el archivo CSV desde {file_path}...")
    if dtypes_dict is not None:
        print("Usando tipos de datos personalizados para las columnas...")
        df = pd.read_csv(file_path, dtype=dtypes_dict, usecols=columns)
    else:
        df = pd.read_csv(file_path, usecols=columns)

    return df


# Función de codificación tipo one-hot encoding utilizando 0/1.
def dummy_encode(df, column):
    """
    Encodes a categorical column into dummy variables using 0/1 encoding.

    Args:
        df: The input DataFrame.
        column: The name of the column to encode.

    Returns:
        A DataFrame with the dummy variables.
    """
    
    # Usar dtype=int para forzar 0/1 desde el inicio
    dummies = pd.get_dummies(df[column], prefix=column, prefix_sep='_', drop_first=True, dtype=int)
    
    return pd.concat([df.drop(column, axis=1), dummies], axis=1)


# Función para codificar por nulo y renombrar la columna agregando '_Missing'
def codificar_por_nulo(df, column):
    """
    Codifica una columna por nulo y renombra la columna agregando '_Missing'.

    Args:
        df: El DataFrame de entrada.
        column: El nombre de la columna a codificar.

    Returns:
        Un DataFrame con la columna codificada y renombrada.
    """

    # Codificamos la actual a 1 para los valores pérdidos
    df[column] = df[column].apply(lambda x: 1 if pd.isna(x) else 0)
    # Renombramos la columna para indicar Missing Value
    new_col_name = column + '_Missing'
    df.rename(columns={column: new_col_name}, inplace=True)


# Función para categorizar compactamente una lista de columnas
def categorizar(df, columnas):
	for col in columnas:
		categorizacion(df, col)

# Función para categorizar una columna
def categorizacion(df, col_name):
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
	
	df = df.drop(f'{col_name}_TE', axis=1)
	df = df.drop(col_name, axis=1)
	df = df.drop(f'{col_name}_grouped', axis=1)


# Función para llenar los valores nulos de una lista de columnas con -1
def llenarNulos(df, cols):
	for col in cols:
		df[col] = df[col].fillna(-1)
		


# ------------------------ Bloque de funciones de agrupación de categorías ------------------------

# Función para agrupar la categrías de la columna 'Census_OSBranch' en solo 4
def agrupar_rama(rama):
    """ Agrupa las categorías de la columna 'Census_OSBranch' en categorías más generales.
    Args:
        rama: El valor de la columna 'Census_OSBranch'.
    Returns:
        Una cadena que representa la categoría general de la rama.

    NOTE: Esta función se usa por medio del método .apply() de pandas.
    """

    if re.match(r'^rs[0-9]+', rama):
        return re.match(r'^rs[0-9]+', rama).group(0)
    elif rama.startswith('rs'):
        return 'rs'
    elif re.match(r'^th[0-9]+', rama):
        return re.match(r'^th[0-9]+', rama).group(0)
    elif rama.startswith('th'):
        return 'th'
    elif rama.startswith('win'):
        return 'win'
    else:
        return 'khmer'
    

# Función para agrupar los canales de 'Census_ActivationChannel' en solo 3
def agrupar_canal(canal):
    """ Agrupa las categorías de la columna 'Census_ActivationChannel' en categorías más generales.
    Args:
        canal: El valor de la columna 'Census_ActivationChannel'.
    Returns:
        Una cadena que representa la categoría general del canal.
    NOTE: Esta función se usa por medio del método .apply() de pandas.
    """

    if canal.startswith('OEM'):
        return 'OEM'
    elif canal.startswith('Retail'):
        return 'Retail'
    else:
        return 'Volume'


def limpiar_smartscreen(df):
    """
    Limpia y estandariza la columna SmartScreen
    Args:
        df (pd.DataFrame): DataFrame original
    Returns:
        pd.DataFrame: DataFrame con SmartScreen limpiado
    """
    print("Limpiando columna SmartScreen...")
    if 'SmartScreen' in df.columns:
        # Convertir a minúsculas para estandarizar
        df['SmartScreen'] = df['SmartScreen'].str.lower()

        # Reemplazar valores inconsistentes
        df['SmartScreen'] = df['SmartScreen'].replace({
            'enabled': 'on',
            'requireadmin': 'requireadmin',
            'promt': 'prompt',
            'promprt': 'prompt',
            'prompt ': 'prompt',  # prompt con espacio
            '0': 'off',
            '00000000': 'off'
        })

        # Rellenar valores nulos
        df['SmartScreen'] = df['SmartScreen'].fillna('NaNNN')

    
def agrupar_valores_poco_representativos(df, columna, umbral=0.02, nombre_categoria='Others'):
    
    frecuencias = df[columna].value_counts(normalize=True)
    categorias_poco_frecuentes = frecuencias[frecuencias < umbral].index
    return df[columna].apply(lambda x: nombre_categoria if x in categorias_poco_frecuentes else x)

def llenar_nulos_con_texto(df, columnas, texto="UNKNOWN"):
    
    print(f"Rellenando nulos en {columnas} con texto: {texto}")
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].fillna(texto)
  

def limpiar_power_platform(df):
    """
    Limpia y estandariza la columna Census_PowerPlatformRoleName
    Args:
        df (pd.DataFrame): DataFrame original
    Returns:
        pd.DataFrame: DataFrame con Census_PowerPlatformRoleName limpiado
    """
    print("Limpiando columna Census_PowerPlatformRoleName...")
    col = 'Census_PowerPlatformRoleName'
    if col in df.columns:
        df[col] = df[col].replace('NaN', 'UNKNOWN').fillna('UNKNOWN')


    print("Limpiando columna Census_ChassisTypeName...")
    col = 'Census_ChassisTypeName'
    if col in df.columns:
        # Reemplazar variaciones de unknown
        df[col] = df[col].fillna('UNKNOWN')

        # Agrupar valores menos comunes en 'Other'
        valores_permitidos = ['Notebook', 'Desktop', 'Laptop', 'UNKNOWN','nan', 'Unknown']
        df[col] = df[col].apply(
            lambda x: x if x in valores_permitidos else 'Other'
        )
