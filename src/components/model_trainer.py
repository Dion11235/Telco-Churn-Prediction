import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import mlflow
from src.utils import generate_roc_curves

from src.logger import logging
from src.exception import CustomException




class modelTrainingConfig:
    best_model_path: str = os.path.join("artifacts", "best_model")
    # local_tracking_uri: str = os.path.join(os.getcwd(), "mlflow_experiments_log")
    local_tracking_uri: str = os.path.join("mlflow_experiments_log")
    experiment_name: str = "telco-churn-classification"
    roc_plot_file_path: str = os.path.join("plots", "roc_curve_all_models.png")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainingConfig()

    def train_and_evaluate_model(self, data_processes, test_X_encoded, test_y_encoded):
        try:
            mlflow.set_tracking_uri(self.model_trainer_config.local_tracking_uri)
            mlflow.set_experiment(self.model_trainer_config.experiment_name)

            model_algorithms = [LogisticRegression(),
                                RandomForestClassifier(),
                                SVC(kernel="rbf", probability=True),
                                XGBClassifier()]
            
            all_fpr = []
            all_tpr = []
            all_auc = []
            experiment_names = []

            self.best_model = None
            self.best_f1_score = -1
            self.best_exp = None

            for data_exp_name, train_x, train_y in data_processes:
                for model in model_algorithms:
                    exp_name = data_exp_name+"-"+model.__class__.__name__
                    with mlflow.start_run(run_name=exp_name):
                        
                        model.fit(train_x, train_y)
                        y_pred = model.predict(test_X_encoded)

                        y_pred_prob = model.predict_proba(test_X_encoded)[:,1]
                        fpr, tpr, _ = roc_curve(test_y_encoded, y_pred_prob)
                        auc_score = auc(fpr, tpr)

                        experiment_names.append(exp_name)
                        all_fpr.append(fpr)
                        all_tpr.append(tpr)
                        all_auc.append(auc_score)

                        mlflow.log_param("data", data_exp_name)
                        mlflow.log_param("model", model.__class__.__name__)

                        current_f1_score = f1_score(test_y_encoded, y_pred)

                        mlflow.log_metric("Accuracy", accuracy_score(test_y_encoded, y_pred))
                        mlflow.log_metric("Precision", precision_score(test_y_encoded, y_pred))
                        mlflow.log_metric("Recall", recall_score(test_y_encoded, y_pred))
                        mlflow.log_metric("F1_score", current_f1_score)
                        
                        mlflow.sklearn.log_model(model, "model")

                        if current_f1_score > self.best_f1_score:
                            self.best_model = model
                            self.best_f1_score = current_f1_score
                            self.best_exp = exp_name

                        logging.info(f"finished experiment {exp_name}. Currently best F1 score : {self.best_f1_score}")
            
            generate_roc_curves(experiment_names=experiment_names,
                                all_fpr=all_fpr,
                                all_tpr=all_tpr,
                                all_auc=all_auc,
                                plot_file_name=self.model_trainer_config.roc_plot_file_path)
            
            mlflow.log_artifact(self.model_trainer_config.roc_plot_file_path, "roc_curves")

            if self.best_f1_score < 0.6:
                logging.info(f"The used models are not good enough; Highest f1-score achieved - {self.best_f1_score}")
                return None
            else:
                logging.info(f"Saving the best model ; best experiment - {self.best_exp} ; highest f1 score - {self.best_f1_score}")
                mlflow.sklearn.save_model(self.best_model, 
                                          self.model_trainer_config.best_model_path, 
                                          serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
                return self.best_model

        except Exception as e:
            raise CustomException(e)