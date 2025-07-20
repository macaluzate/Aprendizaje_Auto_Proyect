# 🛡️ Proyecto Malware Prediction

---

⚠️ Cuidado: El dataset manejado en este proyecto es bastante grande, de aproximadamente 83 columnas y casi 9 millones de filas (registros), por lo cual no se recomienda intentar correrlo localmente, a excepción de que se posea con una máquina con recursos de hardware equiparables a las máquinas de Colab Pro.

## 🚀 Replicación del Proyecto

### 📥 Clonar el Repositorio

Haz clic en el botón `< > Code`, copia la URL HTTPS del repositorio y ejecuta:

```bash
git clone <URL-del-repositorio>
cd <nombre-del-repo>
```

---

### 🐍 Crear y Activar Entorno Virtual

Este proyecto utiliza **Python 3.11**. Asegúrate de tenerlo instalado antes de continuar.

```bash
python3.11 -m venv .venv

# Activar entorno virtual:
# En macOS / Linux:
source .venv/bin/activate

# En Windows:
.\.venv\Scripts\activate
```

Una vez activado, instala las dependencias necesarias:

```bash
pip install pip==24.1.2
pip install -r requirements.txt
```

---

### 🔐 Configurar Variables de Entorno (API de Kaggle)

Este proyecto descarga los datos directamente desde Kaggle, por lo que es necesario configurar tus credenciales de la API:

1. Inicia sesión en tu cuenta de [Kaggle](https://www.kaggle.com/).
2. Inscríbete en la competencia [Microsoft Malware Prediction](https://www.kaggle.com/competitions/microsoft-malware-prediction), para esto te diriges a la seccion `Data` de la competencia, y al final de todo está el boton `Join the competition`
3. Ve a tu perfil → [*Settings*](https://www.kaggle.com/settings).
4. En la sección **API**, genera un nuevo token (botón *Create New Token*).
5. Toma los valores de `"username"` y `"key"` (ya sea los generados en la página o del JSON descargado automáticamente) para crear un archivo `.env` en el directorio raíz del proyecto:

```env
KAGGLE_USERNAME = "miquinterog..."
KAGGLE_KEY = "70ba..."
```

**NOTA:** En caso de tener dificultades, puede elegir descargar manualmente el archivo .zip de la competencia, descomprimir y quedarse con el archivo `train.csv` (archivo usado para el proyecto ya que el de test es exlusivo de la competencia y no cuenta con las etiquetas).
Si elige descargar el archivo, puede omitir en el `main.py` la línea de código de la función `download_dataset()`, aunque esta ya tiene un chequeo condicional para esto.

---

### 📁 Estructura y Archivos del Proyecto

| Archivo                | Descripción                                                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `dtypes_dict.json`     | Diccionario de tipos de datos generado tras leer el dataset completo con `low_memory=False`. Permite lecturas más rápidas posteriormente. |
| `preprocessLibrary.py` | Módulo con funciones para procesamiento modular y reutilizable de los datos.                                                              |
| `main.py`              | Script principal que incluye descarga de datos, preprocesamiento, división de conjuntos, entrenamiento y validación del modelo.           |

---

## 📓 Notebooks Experimentales
- [Análisis Exploratorio de Datos](https://colab.research.google.com/drive/16RhnHyxJNJ1vNOMcJHQFw955vspr3sj-?usp=sharing)
- [Entrenamiento de Logistic Regression, Random Forest y LightGBM](https://colab.research.google.com/drive/1A_TzP07LUeR0PZYHoTH_DV0RnAPSjxTk?usp=sharing)
- [Ejecución con el modelo final y gráfica de métricas](https://colab.research.google.com/drive/1jhhYb3-_4Tm69yi8syuFXEe6G_D5MByH?usp=sharing)
---
