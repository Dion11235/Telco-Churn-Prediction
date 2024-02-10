import os
import sys
from dataclasses import dataclass
import numpy as np

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.logger import logging
from src.exception import CustomException



@dataclass
class DataAugmentationConfig:
    undersampled_train_x_data_enc_path: str = os.path.join("data","augmented", "train_x_oh_encoded_undersampled.npy")
    undersampled_train_y_data_enc_path: str = os.path.join("data","augmented", "train_y_encoded_undersampled.npy")

    oversampled_train_x_data_enc_path: str = os.path.join("data","augmented", "train_x_oh_encoded_oversampled.npy")
    oversampled_train_y_data_enc_path: str = os.path.join("data","augmented", "train_y_encoded_oversampled.npy")

    smote_train_x_data_enc_path: str = os.path.join("data","augmented", "train_x_oh_encoded_smote.npy")
    smote_train_y_data_enc_path: str = os.path.join("data","augmented", "train_y_encoded_smote.npy")



class DataAugmentation:
    def __init__(self):
        self.data_augmentation_configs = DataAugmentationConfig()

    def perform_undersampling(self, X,y, sampling_strategy=1.0, random_state=None):
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)

        return X_resampled, y_resampled


    def perform_oversampling(self, X, y, sampling_strategy=1.0, random_state=None):
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        return X_resampled, y_resampled


    def perform_smote(self, X, y, sampling_strategy=1.0, random_state=None):
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled, y_resampled
    

    def augment_data(self, train_x, train_y):

        try:
            logging.info("Starting Augmentation ...")
            os.makedirs(os.path.dirname(self.data_augmentation_configs.undersampled_train_x_data_enc_path), exist_ok=True)

            # Undersampling
            undersampled_train_x, undersampled_train_y = self.perform_undersampling(train_x, train_y)
            np.save(self.data_augmentation_configs.undersampled_train_x_data_enc_path, undersampled_train_x)
            np.save(self.data_augmentation_configs.undersampled_train_y_data_enc_path, undersampled_train_y)
            logging.info("Undersampling done.")
            unique_values, counts = np.unique(undersampled_train_y, return_counts=True)
            value_counts_dict = dict(zip(unique_values, counts))
            logging.info("\nAfter Undersampling :\n {}".format(value_counts_dict))

            # Oversampling
            oversampled_train_x, oversampled_train_y = self.perform_oversampling(train_x, train_y)
            np.save(self.data_augmentation_configs.oversampled_train_x_data_enc_path, oversampled_train_x)
            np.save(self.data_augmentation_configs.oversampled_train_y_data_enc_path, oversampled_train_y)
            logging.info("Oversampling done.")
            unique_values, counts = np.unique(oversampled_train_y, return_counts=True)
            value_counts_dict = dict(zip(unique_values, counts))
            logging.info("\nAfter Oversampling :\n {}".format(value_counts_dict))

            # SMOTE
            logging.debug(f"{train_x.shape}-{train_y.shape}")
            smote_train_x, smote_train_y = self.perform_smote(train_x, train_y)
            np.save(self.data_augmentation_configs.smote_train_x_data_enc_path, smote_train_x)
            np.save(self.data_augmentation_configs.smote_train_y_data_enc_path, smote_train_y)
            logging.info("SMOTE done.")
            unique_values, counts = np.unique(smote_train_y, return_counts=True)
            value_counts_dict = dict(zip(unique_values, counts))
            logging.info("\nAfter SMOTE :\n {}".format(value_counts_dict))

            return [
                ("original", train_x, train_y),
                ("undersampled", undersampled_train_x, undersampled_train_y),
                ("oversampled", oversampled_train_x, oversampled_train_y),
                ("smoted", smote_train_x, smote_train_y)
            ]
        
        except Exception as e:
            raise CustomException(e)
        

        

        
