from flask import Flask
from flask_cors import CORS

# Routes
from .router import IndexRouter, ModelRouter

app = Flask(__name__)

def init_app(config):
    # Configuration
    app.config.from_object(config)

    # Blueprints
    app.register_blueprint(IndexRouter.main, url_prefix='/')
    app.register_blueprint(ModelRouter.main, url_prefix='/model')

    # Configuración de CORS
    CORS(app, resources={r"*": {"origins": "*"}})
    
    return app