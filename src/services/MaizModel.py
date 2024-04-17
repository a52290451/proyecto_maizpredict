import traceback

# Database
#from src.database.db_mysql import get_connection
# Logger
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.Logger import Logger
# Models
#from src.models.LanguageModel import Language


class MaizModel():

    @classmethod
    def get_model(cls, input_data):
        try:
            model = cls.load_keras_model(input_data['Type'])
            # Realizar predicción
            input_array = np.array([input_data['TMin'], input_data['TMax'], input_data['P'], input_data['PV'], input_data['BSHG']])
            input_normal = cls.normalizacion(input_array)
            final = np.array([[input_normal[0], input_normal[1], input_normal[2], input_normal[3],input_normal[4]]])
            resultado_prediccion = model.predict(final)
            return cls.desnormalizacion(resultado_prediccion)
        except Exception as ex:
            Logger.add_to_log("error", str(ex))
            Logger.add_to_log("error", traceback.format_exc())
    
    @staticmethod
    def normalizacion(variables):
        data = pd.read_csv('src/static/dataset5Mean.csv')
        data = data.drop(columns=['RENDIMIENTO'])
        # Calcular máximos y mínimos para todas las columnas excepto 'BSHG'
        max_values = data.max()
        min_values = data.min()
        # Normalizar cada variable en el arreglo, excepto 'BSHG'
        normalized_variables = []
        cont=0
        for variable in variables:
            max_val = max_values.iloc[cont]
            min_val = min_values.iloc[cont]
            normalized_variable = (variable - min_val) / (max_val - min_val)
            normalized_variables.append(normalized_variable)
            cont+=1
        return normalized_variables
    
    @staticmethod
    def desnormalizacion(resultado):
        data = pd.read_csv('src/static/dataset5Mean.csv')
        # Extrae la columna "RENDIMIENTO"
        rendimiento_column = data['RENDIMIENTO']
        # Encuentra el valor máximo y mínimo
        max_rendimiento = rendimiento_column.max()
        min_rendimiento = rendimiento_column.min()
        resultado_final = min_rendimiento + (max_rendimiento-min_rendimiento)*(resultado)
        resultado_final = resultado_final[0][0]
        return round(resultado_final, 3)  
    
    @staticmethod
    def load_keras_model(type):
        # Define el modelo secuencial con la capa de entrada adecuada
        if (type =="dnn"):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(units=1024, activation='relu', input_shape=(5,)))
            model.add(tf.keras.layers.Dense(units=512, activation='relu'))
            model.add(tf.keras.layers.Dense(units=256, activation='relu'))
            model.add(tf.keras.layers.Dense(units=128, activation='relu'))
            model.add(tf.keras.layers.Dense(units=64, activation='relu'))
            model.add(tf.keras.layers.Dense(units=32, activation='relu'))
            model.add(tf.keras.layers.Dense(units=1, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='Adamax')
            # Carga los pesos del modelo
            model.load_weights('src/static/modelo_DNN_Conf3_v28_95.0.h5')
        elif (type == "cnn"):
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape((5, 1), input_shape=(5,)),
                tf.keras.layers.Conv1D(64, 2, activation='relu'),
                tf.keras.layers.Conv1D(128,2, activation='relu'),
                tf.keras.layers.Conv1D(256, 2, activation='relu'),
                tf.keras.layers.Conv1D(512, 2, activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='Adam', loss='mean_squared_error')
            model.load_weights('src/static/modelo_CNN_Conf8_v3_87.0.h5')
        else:
            model_cnn= tf.keras.Sequential([
                tf.keras.layers.Reshape((5, 1), input_shape=(5,)),
                tf.keras.layers.Conv1D(16, 2, activation='relu'),
                tf.keras.layers.Conv1D(32, 2, activation='relu'),
                tf.keras.layers.Conv1D(64, 2, activation='relu'),
                tf.keras.layers.Conv1D(128, 2, activation='relu'),
                tf.keras.layers.Flatten()
            ])
            model_dnn = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            combined_input = tf.keras.layers.Input(shape=(5,1))
            cnn_output = model_cnn(combined_input)
            dnn_output = model_dnn(cnn_output)
            model = tf.keras.models.Model(inputs=combined_input, outputs=dnn_output)
            model.compile(loss='mean_squared_error', optimizer='Adamax')
            model.load_weights('src/static/modelo_CNN_DNN_ConfA2_v5_87.0.h5')
            
        return model
    
    @classmethod
    def reales_vs_predichos(cls):
        try:
            # Cargar los conjuntos de datos desde archivos CSV
            y_test = pd.read_csv('src/static/modelo_DNN_Conf3_v28_y_test.csv')
            y_pred = pd.read_csv('src/static/modelo_DNN_Conf3_v28_y_pred.csv')

            # Seleccionar los primeros 150 valores
            y_test = y_test.iloc[:150]
            y_pred = y_pred.iloc[:150]
            
            # Convertir los valores a listas
            y_test_list = y_test.values.flatten()
            y_pred_list = [round(value, 3) for value in y_pred.values.flatten()]
            
            # Concatenar los valores en una cadena separada por comas
            y_test_str = ','.join(map(str, y_test_list)).replace('[', '').replace(']', '')
            y_pred_str = ','.join(map(str, y_pred_list)).replace('[', '').replace(']', '')
            
            return y_test_str, y_pred_str
        
        except Exception as ex:
            print("Error cargando los datos:", ex)
            return None, None
        
    @classmethod
    def metricas(cls,urlTest,urlPred,urlTrain,urlPred2):
        
        try:
            # Cargar los conjuntos de datos desde archivos CSV
            y_test = pd.read_csv(urlTest)
            y_pred = pd.read_csv(urlPred)
            y_train = pd.read_csv(urlTrain)
            y_pred2 = pd.read_csv(urlPred2)
            y_test = y_test.values.flatten()
            y_pred = y_pred.values.flatten()
            y_train = y_train.values.flatten()
            y_pred2 = y_pred2.values.flatten()
            # Calcular las métricas validación
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rrmse = rmse / (np.mean(y_test)+0.0000001)
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test+0.0000001))) * 100
            error_pct = np.mean(np.abs((y_test - y_pred) / (y_test+0.0000001))) * 100
            r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
            
            # Calcular las métricas entrenamiento
            rmse_t = np.sqrt(mean_squared_error(y_train, y_pred2))
            rrmse_t = rmse_t / (np.mean(y_train)+0.0000001)
            mae_t = mean_absolute_error(y_train, y_pred2)
            mape_t = np.mean(np.abs((y_train - y_pred2) / (y_train+0.0000001))) * 100
            error_pct_t = np.mean(np.abs((y_train - y_pred2) / (y_train+0.0000001))) * 100
            r2_t = 1 - np.sum((y_train - y_pred2)**2) / np.sum((y_train - np.mean(y_train))**2)
            
            # Retornar las métricas
            return {
                'train':{
                    'rmse': round(rmse_t, 3),
                    'rrmse': round(rrmse_t, 3),
                    'mae': round(mae_t, 3),
                    'mape': round(mape_t, 3),
                    'error_pct': round(error_pct_t, 3),
                    'r2': round(r2_t, 3)
                },
                'test':{
                    'rmse': round(rmse, 3),
                    'rrmse': round(rrmse, 3),
                    'mae': round(mae, 3),
                    'mape': round(mape, 3),
                    'error_pct': round(error_pct, 3),
                    'r2': round(r2, 3)
                }
            }
        except Exception as ex:
            print("Error al calcular las métricas:", ex)
            return None
        
        