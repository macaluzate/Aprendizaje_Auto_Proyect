# Readme

# Proyecto Malware Prediction

## Replicación de este Proyecto

### Ambiente

- Python 3.11
- Instalar las librerías necesarias según el archivo `requirements.txt`
```bash
pip install -r requirements.txt
```

### Preparar variables de entorono
Las únicas variables de entorno necesarias para este proyecto son credenciales de la API de Kaggle para poder descargar programáticamente el archivo.

1. Inicie sesión en su cuenta de [Kaggle](https://www.kaggle.com/)
2. Inscribase a la competencia de [Microsoft Malware](https://www.kaggle.com/competitions/microsoft-malware-prediction)
3. Dirigase a la sección de [*Settings*](https://www.kaggle.com/settings) de su perfil de Kaggle
4. En la subseccción de API genere un nuevo token de acceso (Create New Token).
5. Cree un archivo `.env`en el mismo directorio base del proyecto, con sus credenciales así:
```bash
KAGGLE_USERNAME = "miquinterog..."
KAGGLE_KEY = "70ba..."
```

### Descripción de Archivos

- **preprocessLibrary.py**: Contiene funciones para sintetizar el procesamiento y hacer todo de una manera más modular y sobrecargando menos el archivo principal

- **main.py**: Script principal, que abarca desde la descarga de los datos, preprocesamiento, división en *train* y *test*, hasta el entrenamiento del modelo y su validación en split de test.
