import os
import joblib
import pandas as pd
from src.contexts.api.models import PredictorRequest

class TrainModelController:
    def execute(self, request: PredictorRequest):
        print(f"Petición recibida: {request}")
        
        lr_model_path = os.getenv("MODELO_ENTRENADO", "modelo_genero.joblib")
        
        if not os.path.exists(lr_model_path):
            return {"status": "ERROR", "message": "El modelo no ha sido entrenado o el archivo no existe."}

        # Cargar el pipeline/modelo guardado
        modelo_cargado = joblib.load(lr_model_path)

        # Formatear la entrada en un DataFrame idéntico al de entrenamiento
        nuevo_dato = pd.DataFrame([{
            'tipo_correo': request.tipo_correo,
            'pais': request.pais,
            'ciudad': request.ciudad
        }])

        # Realizar la predicción
        prediccion = modelo_cargado.predict(nuevo_dato)
        genero_predicho = prediccion[0]
        
        print(f"Predicción para el cliente: {genero_predicho}")
        
        return {
            "status": "OK", 
            "resultado": {
                "tipo_correo": request.tipo_correo,
                "pais": request.pais,
                "ciudad": request.ciudad,
                "genero_musical_predicho": genero_predicho
            }
        }
