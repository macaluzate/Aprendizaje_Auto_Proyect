from preprocessLibrary import (
    download_dataset,
    read_malware_csv,
    preprocess_data
)
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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    recall_score, precision_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# Verificamos que el archivo 'train.csv' no exista, si no lo descargamos
if not os.path.exists('train.csv'):
    print("El archivo 'train.csv' no se encuentra. Procediendo a descargar el dataset...")
    # Llamamos a la función para descargar el dataset
    download_dataset()
else:
    print("El archivo 'train.csv' ya existe. No es necesario descargar el dataset.")


# Leamos el .json con el diccionario de los dtypes
with open('dtypes_dict.json', 'r') as f:
    dtypes_dict = json.load(f)

# Columnas a leer del CSV
columns_to_read = ['ProductName', 'EngineVersion', 'AppVersion', 'AvSigVersion',
            'IsBeta', 'RtpStateBitfield', 'IsSxsPassiveMode', 'DefaultBrowsersIdentifier',
            'AVProductsInstalled', 'AVProductsEnabled', 'HasTpm',
            'CountryIdentifier', 'CityIdentifier', 'OrganizationIdentifier', 'GeoNameIdentifier',
            'LocaleEnglishNameIdentifier', 'Platform', 'Processor', 'OsVer', 'OsBuild', 'OsSuite',
            'OsPlatformSubRelease','OsBuildLab', 'SkuEdition', 'IsProtected', 'AutoSampleOptIn',
            'SMode', 'SmartScreen', 'Firewall', 'UacLuaenable', 'Census_MDC2FormFactor', 'Census_DeviceFamily', 'Census_OEMNameIdentifier',
            'Census_OEMModelIdentifier', 'Census_ProcessorCoreCount', 'Census_ProcessorManufacturerIdentifier',
            'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName', 'Census_SystemVolumeTotalCapacity', 'Census_TotalPhysicalRAM',
            'Census_ChassisTypeName', 'Census_InternalPrimaryDiagonalDisplaySizeInInches', 'Census_InternalPrimaryDisplayResolutionHorizontal',
            'Census_InternalPrimaryDisplayResolutionVertical', 'Census_PowerPlatformRoleName',
            'Census_OSArchitecture', 'Census_OSBranch', 'Census_OSWUAutoUpdateOptionsName',
            'Census_IsPortableOperatingSystem', 'Census_GenuineStateName',
            'Census_ActivationChannel', 'Census_IsFlightingInternal',
            'Census_FirmwareManufacturerIdentifier', 'Census_IsSecureBootEnabled',
            'Census_IsWIMBootEnabled', 'Census_IsVirtualDevice', 'Census_IsTouchEnabled',
            'Census_IsPenCapable', 'Census_IsAlwaysOnAlwaysConnectedCapable','Wdft_IsGamer',
            # Variable objetivo
            'HasDetections'
]

# Leemos el archivo CSV para obtener el DataFrame de pandas
df = read_malware_csv(file_path='train.csv', dtypes_dict=dtypes_dict,
                          columns=columns_to_read)

# Dividimos en 80% entrenamiento y 20% prueba, manteniendo HasDetections
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    # La hacemos estratificada para mantener la proporción de la variable objetivo
    # (aunque en el EDA se comprobó que están balanceadas las clases)
    stratify=df["HasDetections"]
)


train_df_proc = preprocess_data(train_df)
test_df_proc = preprocess_data(test_df)

# 1. Definimos las columnas para hacer OneHotEncoding y TargetEncoding
onehot_cols = ['Census_OSArchitecture', 'Census_OSBranch',
                        'Census_OSWUAutoUpdateOptionsName',
                        'Census_ActivationChannel', 'ProductName',
                        'SkuEdition','Census_PrimaryDiskTypeName',
                        'Census_PowerPlatformRoleName', 'ProductName',
                        'Platform','Processor','OsPlatformSubRelease',
                        'SkuEdition', 'Census_MDC2FormFactor',
                        'Census_DeviceFamily', 'Census_PrimaryDiskTypeName',
                        'Census_PowerPlatformRoleName','Census_OSArchitecture',
                        'Census_OSBranch','Census_OSWUAutoUpdateOptionsName',
                        'Census_ActivationChannel']
target_cols = ['EngineVersion','AppVersion', 'AvSigVersion','DefaultBrowsersIdentifier', 'CountryIdentifier', 'CityIdentifier', 'GeoNameIdentifier', 'OrganizationIdentifier', 'OsVer', 'OsBuild', 'OsSuite', 'OsBuildLab', 'Census_FirmwareManufacturerIdentifier', 'SmartScreen', 'Census_ChassisTypeName']

# 2. Dividimos las features y el target
X_train = train_df_proc.drop(columns='HasDetections')
y_train = train_df_proc['HasDetections']
X_test = test_df_proc.drop(columns='HasDetections')
y_test = test_df_proc['HasDetections']

# 3. Rellena los nulos antes
for col in onehot_cols + target_cols:
    X_train[col] = X_train[col].fillna('Missing')
    X_test[col] = X_test[col].fillna('Missing')


# 4. Preprocesamiento: combinación de OneHot y TargetEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        #('target', TargetEncoder(cols=target_cols), target_cols)
        ('target', TargetEncoder(smooth="auto"), target_cols)
    ],
    remainder='passthrough'
)

# 5. Construímos pipeline con el modelo
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LGBMClassifier(random_state=42))
])

# 6. Entrenamos con X_train, y_train
pipeline.fit(X_train, y_train)

# 7. Realizamos predicciones sobre el conjunto de prueba y calculamos métricas de evaluación
y_pred = pipeline.predict(X_test) # Etiquetas predichas
y_proba = pipeline.predict_proba(X_test)[:, 1] # Probabilidades para la clase positiva (1)


print("Métricas de evaluación en entrenamiento:")
y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_train_proba)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
print("AUC:", train_auc)
print("Accuracy:", train_accuracy)
print("F1 Score:", train_f1)
print("Recall:", train_recall)
print("Precision:", train_precision)

print('----'*20)
print("Métricas de evaluación en test:")
auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print("AUC:", auc)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Recall:", recall)
print("Precision:", precision)