import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from dotenv import load_dotenv
import os
import zipfile
import pandas as pd


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
        train_df = pd.read_csv(file_path, dtype=dtypes_dict, usecols=columns)
    else:
        train_df = pd.read_csv(file_path, usecols=columns)

    print("Archivo CSV leído correctamente.")
    return train_df




# ------------------------ Funciones utilitarias ------------------------

def llenar_nulos_con_texto(df, columnas, texto="UNKNOWN"):
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].fillna(texto)

def llenarNulos(train_df, columnas):
    for col in columnas:
        if col in train_df.columns:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                train_df[col] = train_df[col].fillna(-1)
            else:
                train_df[col] = train_df[col].fillna('Missing')


def imputar_con_media_y_marcar_nulos(df, columnas):
    for col in columnas:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            # df[col].fillna(df[col].mean(), inplace=True)
            df[col] = df[col].fillna(df[col].mean())


# ------------------------ Agrupación y limpieza ------------------------

def limpiar_smartscreen(train_df):
    if 'SmartScreen' in train_df.columns:
        train_df['SmartScreen'] = (
            train_df['SmartScreen']
            .str.lower()
            .replace({
                'enabled': 'on', 'requireadmin': 'requireadmin',
                'promt': 'prompt', 'promprt': 'prompt', 'prompt ': 'prompt',
                '0': 'off', '00000000': 'off'
            })
            .fillna('NaNNN')
        )

def limpiar_power_platform(train_df):
    if 'Census_PowerPlatformRoleName' in train_df.columns:
        train_df['Census_PowerPlatformRoleName'] = train_df['Census_PowerPlatformRoleName'].replace('NaN', 'UNKNOWN').fillna('UNKNOWN')
    if 'Census_ChassisTypeName' in train_df.columns:
        valores_permitidos = ['Notebook', 'Desktop', 'Laptop', 'UNKNOWN', 'nan', 'Unknown']
        train_df['Census_ChassisTypeName'] = train_df['Census_ChassisTypeName'].fillna('UNKNOWN').apply(
            lambda x: x if x in valores_permitidos else 'Other')

def agrupar_valores_poco_representativos(train_df, columna, umbral=0.02, nombre_categoria='Others'):
    if columna in train_df.columns:
        frecuencias = train_df[columna].value_counts(normalize=True)
        categorias_poco_frecuentes = frecuencias[frecuencias < umbral].index
        train_df[columna] = train_df[columna].apply(lambda x: nombre_categoria if x in categorias_poco_frecuentes else x)

def agrupar_rama(rama):
    if isinstance(rama, str):
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
    return 'khmer'

def agrupar_canal(canal):
    if isinstance(canal, str):
        if canal.startswith('OEM'):
            return 'OEM'
        elif canal.startswith('Retail'):
            return 'Retail'
    return 'Volume'



def preprocess_data(train_df):
  llenarNull = ['RtpStateBitfield', 'DefaultBrowsersIdentifier', 'CityIdentifier', 'GeoNameIdentifier','OrganizationIdentifier', 'OsBuildLab', 'SMode',
          'Firewall',
          'UacLuaenable',
          'Census_TotalPhysicalRAM',
          'Census_InternalPrimaryDiagonalDisplaySizeInInches',
          'Census_InternalPrimaryDisplayResolutionVertical',
          'Census_InternalPrimaryDisplayResolutionHorizontal',
          'Census_IsFlightingInternal', 'Census_FirmwareManufacturerIdentifier',
          'Census_IsWIMBootEnabled', 'Census_IsVirtualDevice',
          'Census_IsAlwaysOnAlwaysConnectedCapable',
          'Wdft_IsGamer', 'Census_IsFlightingInternal','Census_FirmwareManufacturerIdentifier', 'Census_IsWIMBootEnabled']

  columnas_nulos_texto = ['Census_PrimaryDiskTypeName']
  # Aplicar limpieza modular usando las funciones importadas
  llenar_nulos_con_texto(train_df, columnas_nulos_texto, "UNKNOWN")

  llenarNulos(train_df, llenarNull)
  limpiar_smartscreen(train_df)
  agrupar_valores_poco_representativos(train_df, 'SmartScreen')
  # train_df['SmartScreen'] = agrupar_valores_poco_representativos(train_df, 'SmartScreen', umbral=0.02, nombre_categoria='Others')
  # df = limpiar_chassis_type(df)
  limpiar_power_platform(train_df)

  # configs = categorizar(train_df, cat)

  # Otros procesamiento individuales
  train_df['Census_OSBranch'] = train_df['Census_OSBranch'].apply(agrupar_rama)
  train_df['Census_GenuineStateName'] = train_df['Census_GenuineStateName'].apply(lambda x: 0 if x == 'IS_GENUINE' else 1)
  train_df['Census_ActivationChannel'] = train_df['Census_ActivationChannel'].apply(agrupar_canal)

  # Columnas para aplicar One-Hot Encoding
  # columns_to_one_hot = []

  # Aplicamos One-Hot Encoding a las columnas especificadas
  # for col in columns_to_one_hot:
      # train_df = dummy_encode(train_df, col)

  imputar_con_media_y_marcar_nulos(train_df, ['AVProductsInstalled','AVProductsEnabled'])

  return train_df