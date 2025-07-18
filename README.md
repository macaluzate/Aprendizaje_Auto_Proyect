# 🛡️ Proyecto Malware Prediction

---

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
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 🔐 Configurar Variables de Entorno (API de Kaggle)

Este proyecto descarga los datos directamente desde Kaggle, por lo que es necesario configurar tus credenciales de la API:

1. Inicia sesión en tu cuenta de [Kaggle](https://www.kaggle.com/).
2. Inscríbete en la competencia [Microsoft Malware Prediction](https://www.kaggle.com/competitions/microsoft-malware-prediction).
3. Ve a tu perfil → [*Settings*](https://www.kaggle.com/settings).
4. En la sección **API**, genera un nuevo token (botón *Create New Token*).
5. Se descargará un archivo `kaggle.json`. Toma los valores de `"username"` y `"key"` para crear un archivo `.env` en el directorio raíz del proyecto:

```env
KAGGLE_USERNAME=miquinterog...
KAGGLE_KEY=70ba...
```

---

### 📁 Estructura y Archivos del Proyecto

| Archivo                | Descripción                                                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `dtypes_dict.json`     | Diccionario de tipos de datos generado tras leer el dataset completo con `low_memory=False`. Permite lecturas más rápidas posteriormente. |
| `preprocessLibrary.py` | Módulo con funciones para procesamiento modular y reutilizable de los datos.                                                              |
| `main.py`              | Script principal que incluye descarga de datos, preprocesamiento, división de conjuntos, entrenamiento y validación del modelo.           |

---

## 📓 Notebooks Experimentales

En esta sección puedes incluir o enlazar notebooks donde hayas probado técnicas adicionales o exploraciones fuera del flujo principal.

---

¿Quieres que también te ayude a escribir una breve introducción general del proyecto (problema, objetivo, datos)?
