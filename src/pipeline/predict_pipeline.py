import os
import mlflow
from src.utils import load_object, load_json
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class InferenceConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    model_path: str = os.path.join("artifacts", "best_model")
    class_labels_path: str = os.path.join("data", "encoded", "class_encodings.json")
    
class ChurnInference:
    def __init__(self):
        try:
            self.inference_config = InferenceConfig()
            self.preprocessor = load_object(self.inference_config.preprocessor_path)
            self.model = mlflow.sklearn.load_model(self.inference_config.model_path)
            self.class_labels = load_json(self.inference_config.class_labels_path)
            logging.info("preprocessor and model objects are loaded.")
        
        except Exception as e:
            raise CustomException(e)
        
        
    def predict_churn(self, df):
        try:
            processed_df = self.preprocessor.transform(df)
            logging.info("data processed ...")
            predictions = self.model.predict(processed_df)
            prediction_confs = self.model.predict_proba(processed_df)
            logging.info("prediction complete ...")
            enc_to_class = {v:k for k,v in self.class_labels.items()}
            df["predicted_churn"] = [enc_to_class[c] for c in predictions]
            df["prediction_confidence"] = [prediction_confs[:,c] for c in predictions]
            
            return df
        
        except Exception as e:
            raise CustomException(e)
    

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(os.path.join("data", "intermediate", "test_x.csv"))
    
    churn_pred_model = ChurnInference()
    predicted_df = churn_pred_model.predict_churn(df.iloc[:10,:])
    
    predicted_df.to_csv("dummy_prediction.csv", index=False)
    