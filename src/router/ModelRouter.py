from flask import Blueprint, jsonify, request

import traceback

# Logger
from src.utils.Logger import Logger

# Services
from src.services.MaizModel import MaizModel

main = Blueprint('model_blueprint', __name__)


@main.route('/prediccion', methods=['POST'])
def get_prediccion():
    
    # Obtener datos de entrada de la solicitud (asumiendo que los valores de entrada se envían como JSON)
    
    try:
        print("LLEGAAAAAAAAAAAAAAAAAAAAA")
        input_data = request.json
        # Verificar si se proporcionaron datos de entrada
        print(input_data)
        
        print("..................................")
        if input_data:
            # Llamar al método get_model con los datos de entrada y obtener el resultado de la predicción
            resultado_prediccion = MaizModel.get_model(input_data)
            return jsonify({'prediction': float("{:.3f}".format(resultado_prediccion)), 'message': "SUCCESS", 'success': True}), 200
        else:
            return jsonify({'message': "NO_INPUT_DATA_PROVIDED", 'success': False}), 400
    except Exception as ex:
        Logger.add_to_log("error", str(ex))
        Logger.add_to_log("error", traceback.format_exc())
        return jsonify({'message': "ERROR", 'success': False}), 400
    
@main.route('/comparacion', methods=['GET'])
def get_comparacion():
    try:
        y_test, y_pred = MaizModel.reales_vs_predichos()   
        if y_test is not None and y_pred is not None:
            # Devolver los datos cargados como JSON
            return jsonify({
                'y_test': y_test,
                'y_pred': y_pred,
                'message': "Data loaded successfully",
                'success': True
            }), 200
        else:
            return jsonify({'message': "Failed to load data", 'success': False}), 400
    except Exception as ex:
        Logger.add_to_log("error", str(ex))
        Logger.add_to_log("error", traceback.format_exc())
        return jsonify({'message': "ERROR", 'success': False}), 400

@main.route('/metricas', methods=['GET'])
def get_metricas():
    try:
        # Obtener las métricas utilizando el método definido en la clase MaizModel
        metrics_DNN = MaizModel.metricas('src/static/modelo_DNN_Conf3_v28_y_test.csv',
                                     'src/static/modelo_DNN_Conf3_v28_y_pred.csv',
                                     'src/static/modelo_DNN_Conf3_v28_y_train.csv',
                                     'src/static/modelo_DNN_Conf3_v28_y_pred2.csv')
        metrics_CNN = MaizModel.metricas('src/static/modelo_CNN_Conf8_v3_y_test.csv',
                                     'src/static/modelo_CNN_Conf8_v3_y_pred.csv',
                                     'src/static/modelo_CNN_Conf8_v3_y_train.csv',
                                     'src/static/modelo_CNN_Conf8_v3_y_pred2.csv')
        metrics_CNN_DNN = MaizModel.metricas('src/static/modelo_CNN_DNN_ConfA2_v5_y_test.csv',
                                     'src/static/modelo_CNN_DNN_ConfA2_v5_y_pred.csv',
                                     'src/static/modelo_CNN_DNN_ConfA2_v5_y_train.csv',
                                     'src/static/modelo_CNN_DNN_ConfA2_v5_y_pred2.csv')
        if metrics_DNN and metrics_CNN and metrics_CNN_DNN:
            # Devolver las métricas como JSON
            merged_metrics = {
                'DNN': metrics_DNN,
                'CNN': metrics_CNN,
                'CNN_DNN': metrics_CNN_DNN
            }
            
            # Devolver las métricas combinadas como JSON
            return jsonify({
                'metrics': merged_metrics,
                'message': "Metrics calculated successfully",
                'success': True
            }), 200
        else:
            return jsonify({'message': "Failed to calculate metrics", 'success': False}), 400
    except Exception as ex:
        Logger.add_to_log("error", str(ex))
        Logger.add_to_log("error", traceback.format_exc())
        return jsonify({'message': "ERROR", 'success': False}), 400