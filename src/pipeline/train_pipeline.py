from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_augmentation import DataAugmentation
from src.components.model_trainer import ModelTrainer


# data ingestion - loading and splitting data 
data_inj = DataIngestion()
train_x_path, train_y_path, test_x_path, test_y_path = data_inj.ingest_data()


# data transformation - transforming data (Scaling, One-hot-encoding)
data_transformation = DataTransformation()
train_x_enc, train_y_enc, test_x_enc, test_y_enc, preprocessor_path = data_transformation.transform_data(train_x_path,
                                                                                                        train_y_path,
                                                                                                        test_x_path,
                                                                                                        test_y_path)
# data augmentation - augmentation to tackle imbalance in the data (Undersampling, Oversampling, SMOTE)
data_augmentation = DataAugmentation()
augmented_data_list = data_augmentation.augment_data(train_x_enc, train_y_enc)

# model training - Selecting the best model with experiment tracking using MLflow
model_trainer = ModelTrainer()
best_model = model_trainer.train_and_evaluate_model(augmented_data_list, test_x_enc, test_y_enc)