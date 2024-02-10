import os
import sys

print("Current directory:\n", os.getcwd(),"\n\n")

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass


"""
# all about dataclass
from dataclasses import dataclass

@dataclass
class Movie:
    title: str
    genre: str
    release_year: int
    rating: float
    director: str
    # Add other relevant attributes as needed

# Example instances
movie1 = Movie(title="Inception", genre="Sci-Fi", release_year=2010, rating=8.8, director="Christopher Nolan")
movie2 = Movie(title="The Shawshank Redemption", genre="Drama", release_year=1994, rating=9.3, director="Frank Darabont")

# __init__ method
# Automatically generated
# Initializes the object with provided arguments
# No need to explicitly define this method

# __repr__ method
# Automatically generated
# Generates a string representation of the object for debugging purposes
print(repr(movie1))  # Output: Movie(title='Inception', genre='Sci-Fi', release_year=2010, rating=8.8, director='Christopher Nolan')

# __eq__ method
# Automatically generated
# Implements equality comparison based on attribute values
print(movie1 == movie2)  # Output: False

# __ne__ method
# Automatically generated
# Implements inequality comparison (negation of __eq__)
print(movie1 != movie2)  # Output: True

# __lt__, __le__, __gt__, __ge__ methods
# Automatically generated
# Implement rich comparison methods based on attribute values
print(movie1 < movie2)  # Output: True

# __hash__ method
# Automatically generated
# Generates a hash value for the object, useful for using instances in hashed collections
hash_value = hash(movie1)

# asdict() method
# Automatically generated
# Returns the object's attributes as a dictionary
movie_dict = movie1.__dict__

# astuple() method
# Automatically generated
# Returns the object's attributes as a tuple
movie_tuple = movie1.__tuple__

"""


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data","raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    intermediate_raw_data_path: str = os.path.join("data","intermediate", "intermediate_data.csv")

    train_x_data_path: str = os.path.join("data","intermediate", "train_x.csv")
    train_y_data_path: str = os.path.join("data","intermediate", "train_y.csv")

    test_x_data_path: str = os.path.join("data","intermediate", "test_x.csv")
    test_y_data_path: str = os.path.join("data","intermediate", "test_y.csv")



class DataIngestion:
    def __init__(self):
        self.data_configs = DataIngestionConfig()

    def change_dtype(self, value, to_type=float):
        try:
            return to_type(value)
        except:
            None
        
    def ingest_data(self):
        logging.info("Initiated data ingestion ...")
        try:
            # data loading 
            df = pd.read_csv(self.data_configs.raw_data_path)
            logging.info("Loaded the raw data ...")
            

            # dtype fixing and NA handling
            df["TotalCharges"] = df["TotalCharges"].apply(lambda x: self.change_dtype(x, float))
            df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: self.change_dtype(x, str))
            df.dropna(inplace=True)

            # saving intermediate data
            os.makedirs(os.path.dirname(self.data_configs.intermediate_raw_data_path), exist_ok=True)
            df.to_csv(self.data_configs.intermediate_raw_data_path, index=False)
            logging.info(f"Data type fixing is done and saved at {self.data_configs.intermediate_raw_data_path}...")

            # categorical, numeric features and target columns
            self.categorical_cols = [col for col in df.columns if df[col].dtypes == object and col not in ["customerID", "Churn"]]
            self.numeric_cols = [col for col in df.columns if df[col].dtypes != object and col not in ["customerID", "Churn"]]
            self.target_col = ["Churn"]

            # train test split
            X_df = df[self.categorical_cols+self.numeric_cols]
            y_df = df[self.target_col]
            self.train_X_df, self.test_X_df, self.train_y_df, self.test_y_df = train_test_split(X_df, y_df, 
                                                                                                test_size=0.2, 
                                                                                                stratify=y_df,
                                                                                                random_state=2)

            # saving train x,y dataframes
            os.makedirs(os.path.dirname(self.data_configs.train_x_data_path), exist_ok=True)
            self.train_X_df.to_csv(self.data_configs.train_x_data_path, index=False)
            self.train_y_df.to_csv(self.data_configs.train_y_data_path, index=False)

            # saving test x,y dataframes
            self.test_X_df.to_csv(self.data_configs.test_x_data_path, index=False)
            self.test_y_df.to_csv(self.data_configs.test_y_data_path, index=False)

            

            logging.info(f"data splitting are done and saved in the folder {os.path.dirname(self.data_configs.train_x_data_path)}")
            logging.info("Training data size : {} {}".format(len(self.train_X_df), len(self.train_y_df)))
            logging.info("Test data size : {} {}".format(len(self.test_X_df), len(self.test_y_df)))

            logging.info("in Train data :\n{}".format(self.train_y_df.value_counts()/self.train_y_df.value_counts().sum()))
            logging.info("in Test data :\n{}".format(self.test_y_df.value_counts()/self.test_y_df.value_counts().sum()))

            logging.info("DATA INGESTION DONE ... !!!")

            return(
                self.data_configs.train_x_data_path,
                self.data_configs.train_y_data_path,
                self.data_configs.test_x_data_path,
                self.data_configs.test_y_data_path
            )
        
        except Exception as e:
            raise CustomException(e)



if __name__ == "__main__":
    data_inj = DataIngestion()
    train_x_path, train_y_path, test_x_path, test_y_path = data_inj.ingest_data()