import os
import joblib
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class TrainModel:

    @staticmethod
    def entrenarModelo():
        load_dotenv("/app/.env")
        USER = os.getenv("SUPABASE_USER")
        PASSWORD = os.getenv("SUPABASE_PASSWORD")
        HOST = os.getenv("SUPABASE_HOST")
        PORT = os.getenv("SUPABASE_PORT")
        DBNAME = os.getenv("SUPABASE_DBNAME")
        
        if PORT is None:
            print("No se lee el archivo .env")
            return
        else:
            print("Se lee correctamente el .env")

        try:
            with psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            ) as connection:
                # Cargar vista directamente a un DataFrame de Pandas
                query = 'SELECT tipo_correo, pais, ciudad, genero_musical FROM vista_cliente_genero;'
                df = pd.read_sql_query(query, connection)
                print(f"Filas recuperadas: {len(df)}")

        except Exception as e:
            print(f"Error al conectar o recuperar datos: {e}")
            return
        
        if df.empty:
            print("No se recuperaron filas de la base de datos. Abortando entrenamiento.")
            return
        
        print("Muestra de datos recuperados:")
        print(df.head(2))

        # Separar variables predictoras (X) y variable objetivo (y)
        X = df[['tipo_correo', 'pais', 'ciudad']]
        y = df['genero_musical']

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocesamiento de variables categóricas
        categorical_features = ['tipo_correo', 'pais', 'ciudad']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Crear el Pipeline con Preprocesamiento + Modelo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Entrenar el pipeline completo
        pipeline.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = pipeline.predict(X_test)
        print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))

        # Guardar el pipeline entrenado
        model_path = os.getenv("MODELO_ENTRENADO", "modelo_genero.joblib")
        joblib.dump(pipeline, model_path)
        print(f"Modelo entrenado y guardado exitosamente en: {model_path}")
