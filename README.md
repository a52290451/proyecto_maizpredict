# **Proyecto Flask**

Este proyecto utiliza Flask como framework principal y está diseñado para ejecutarse en un entorno virtual de Python.

---

## **Requisitos previos**

Asegúrate de tener instalado lo siguiente:

1. **Python 3.7 o superior**: Puedes descargarlo desde [python.org](https://www.python.org/).
2. **Pip**: El gestor de paquetes de Python (se incluye automáticamente con Python 3.4+).
3. **Git** (opcional): Para clonar este repositorio.

---

## **Pasos para ejecutar el proyecto**

### 1. Clonar el repositorio
Si el proyecto está en un repositorio, clónalo utilizando Git:
```bash
git clone https://github.com/a52290451/proyecto_maizpredict.git
cd proyecto_maizpredict
```

### 2. Crear y activar un entorno virtual
Crea un entorno virtual para evitar conflictos con otras dependencias.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
Instala las dependencias necesarias desde el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Configura la variable FLASK_APP para que Flask pueda ubicar el archivo principal del proyecto:

**Windows:**
```bash
set FLASK_APP=index.py
```

**macOS/Linux:**
```bash
export FLASK_APP=main.py
```

Para habilitar el modo de desarrollo (opcional):

**Windows:**
```bash
set FLASK_ENV=development
```

**macOS/Linux:**
```bash
export FLASK_ENV=development
```

### 5. Ejecutar el proyecto
Inicia el servidor Flask con el siguiente comando:
```bash
flask run
```