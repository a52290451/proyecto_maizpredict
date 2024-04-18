import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Configura el nivel de registro de TensorFlow a '2' (oculta mensajes de información y advertencias)

from config import config
from src import init_app

configuration = config['development']
app = init_app(configuration)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)