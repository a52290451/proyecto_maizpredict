from flask import Blueprint, jsonify, request

import traceback

# Logger
from src.utils.Logger import Logger

# Services
from src.services.MaizModel import MaizModel

main = Blueprint('index_blueprint', __name__)


@main.route('/')
def index():
    try:
        Logger.add_to_log("info", "{} {}".format(request.method, request.path))
        return "Ok"
    except Exception as ex:
        Logger.add_to_log("error", str(ex))
        Logger.add_to_log("error", traceback.format_exc())

        response = jsonify({'message': "Internal Server Error", 'success': False})
        return response, 500

@main.route('/maizmodel/')
def get_languages():
    try:
        languages = MaizModel.get_model()
        if (len(languages) > 0):
            return jsonify({'languages': languages, 'message': "SUCCESS", 'success': True})
        else:
            return jsonify({'message': "NOTFOUND", 'success': True})
    except Exception as ex:
        Logger.add_to_log("error", str(ex))
        Logger.add_to_log("error", traceback.format_exc())

        return jsonify({'message': "ERROR", 'success': False})