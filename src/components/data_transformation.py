import sys
import os
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import json


@dataclass
class DataTransformationConfig:
    train_x_data_enc_path: str = os.path.join("data","encoded", "train_x_oh_encoded.npy")
    train_y_data_enc_path: str = os.path.join("data","encoded", "train_y_encoded.npy")

    test_x_data_enc_path: str = os.path.join("data","encoded", "test_x_oh_encoded.npy")
    test_y_data_enc_path: str = os.path.join("data","encoded", "test_y_encoded.npy")

    gt_enc_path: str = os.path.join("data","encoded", "class_encodings.json")

    preprocessor_obj_path: str = os.path.join("artifacts","preprocessor", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
         self.data_transform_configs = DataTransformationConfig()

    def get_data_transformer_obj(self, train_x):
        try:
            self.categorical_cols = [col for col in train_x.columns if train_x[col].dtypes == object]
            self.numeric_cols = [col for col in train_x.columns if train_x[col].dtypes != object]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipleline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder())
                ]
            )

            self.preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, self.numeric_cols),
                    ("categorical_pipeline", cat_pipleline, self.categorical_cols)
                ]
            )

            return self.preprocessor

        except Exception as e:
            raise CustomException(e)
        

    def transform_data(self, train_x_path, train_y_path, test_x_path, test_y_path):
        try:
            train_x = pd.read_csv(train_x_path)
            train_y = pd.read_csv(train_y_path)
            test_x = pd.read_csv(test_x_path)
            test_y = pd.read_csv(test_y_path)
        
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj(train_x)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            self.class_labels = {c:i for i,c in enumerate(set(train_y["Churn"]))}

            self.train_X_encoded = preprocessing_obj.fit_transform(train_x)
            self.train_y_encoded = train_y["Churn"].map(self.class_labels).values
            self.test_X_encoded = preprocessing_obj.transform(test_x)
            self.test_y_encoded = test_y["Churn"].map(self.class_labels).values
            

            logging.info(f"Saving preprocessing objects and encoded datasets...")

            # saving train x,y encoded arrays
            os.makedirs(os.path.dirname(self.data_transform_configs.train_x_data_enc_path), exist_ok=True)
            np.save(self.data_transform_configs.train_x_data_enc_path, self.train_X_encoded)
            np.save(self.data_transform_configs.train_y_data_enc_path, self.train_y_encoded)

            # saving test x,y encoded arrays
            np.save(self.data_transform_configs.test_x_data_enc_path, self.test_X_encoded)
            np.save(self.data_transform_configs.test_y_data_enc_path, self.test_y_encoded)

            # saving class labellings
            with open(self.data_transform_configs.gt_enc_path, "w") as f:
                json.dump(self.class_labels, f)

            save_object(

                file_path=self.data_transform_configs.preprocessor_obj_path,
                obj=preprocessing_obj

            )
            logging.info("DATA TRANSFORMATION DONE ... !!!")

            return (
                    self.train_X_encoded,
                    self.train_y_encoded,
                    self.test_X_encoded,
                    self.test_y_encoded,
                    self.data_transform_configs.preprocessor_obj_path

            )


        except Exception as e:
                raise CustomException(e)
                
