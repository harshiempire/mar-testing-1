from ts.torch_handler.base_handler import BaseHandler
import joblib
import pandas as pd
import os

class CarPricePredictionHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        print(f"Model directory: {model_dir}")

        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        ohe_path = os.path.join(model_dir, 'ohe.joblib')
        model_path = os.path.join(model_dir, 'XGBoost.joblib')
        print(f"Scaler path: {scaler_path}")
        print(f"OHE path: {ohe_path}")
        print(f"Model path: {model_path}")

        self.scaler = joblib.load(scaler_path)
        self.ohe = joblib.load(ohe_path)
        self.model = joblib.load(model_path)
        self.initialized = True
        print("Initialization complete.")

    def preprocess(self, data):
        inputs = data[0].get("body")
        row = [
            inputs.get('fuel_type'), inputs.get('aspiration'), inputs.get('door_number'),
            inputs.get('car_body'), inputs.get('drive_wheel'), inputs.get('engine_location'),
            inputs.get('wheelbase'), inputs.get('carlength'), inputs.get('carwidth'),
            inputs.get('carheight'), inputs.get('curbweight'), inputs.get('engine_type'),
            inputs.get('cylinder_number'), inputs.get('enginesize'), inputs.get('fuel_system'),
            inputs.get('boreratio'), inputs.get('stroke'), inputs.get('compression_ratio'),
            inputs.get('horsepower'), inputs.get('peakrpm'), inputs.get('citympg'), inputs.get('highwaympg')
        ]
        print(f"Preprocessed row: {row}")
        return row

    def inference(self, row):
        df = pd.DataFrame([row], columns=[
            'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
            'enginelocation', 'wheelbase', 'carlength', 'carwidth', 'carheight',
            'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'fuelsystem',
            'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
            'citympg', 'highwaympg'
        ])
        print(f"DataFrame before processing: \n{df}")

        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        print(f"Scaled numeric columns: \n{df[numeric_cols]}")

        cat_cols = df.select_dtypes(exclude=['float', 'int']).columns
        df_ohe = pd.DataFrame(self.ohe.transform(df[cat_cols]), columns=self.ohe.get_feature_names_out(cat_cols))
        print(f"One-hot encoded columns: \n{df_ohe}")

        df = pd.concat([df[numeric_cols], df_ohe], axis=1)
        print(f"DataFrame after processing: \n{df}")

        price = self.model.predict(df)[0]
        print(f"Predicted price: {price}")
        return price

    def postprocess(self, inference_output):
    # Ensure the output is JSON serializable by converting it to a regular float
        result = [{"predicted_price": float(inference_output)}]
        print(f"Postprocessed output: {result}")
        return result