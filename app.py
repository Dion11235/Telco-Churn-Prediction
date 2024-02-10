import streamlit as st
import os
import pandas as pd
from src.pipeline.predict_pipeline import ChurnInference
from src.utils import load_json
from dataclasses import dataclass

@dataclass
class AppConfigs:
    column_values_path:str = os.path.join("data","encoded","column_unique_values.json")
    
    
class ChurnApp:
    def __init__(self):
        self.app_configs = AppConfigs()
        self.input_sections = load_json(self.app_configs.column_values_path)
        self.model = ChurnInference()
        
    def display_prediction(self, prediction, probability):
        st.write("## Prediction:")
        if prediction == "Yes":
            st.write(f"The customer will be churned.")
        else:
            st.write(f"The customer will not be churned.")

        # Display probability as a confidence bar
        confidence = probability * 100
        st.write(f"predicted with a {confidence[0]:.2f}% confidence.")
        st.progress(probability[0])
        
    def run_app(self):
        st.set_page_config(layout="wide")
        
        st.markdown("<h1 style='color: cyan;'>Telco Churn Classification Web App</h1>", unsafe_allow_html=True)
        st.caption("For the GitHub code, [click here](https://github.com/Dion11235/Telco-Churn-Prediction)")

        result_columns = st.columns([0.7,0.3], gap="medium")  # Create 1 column for predictions

        with result_columns[0]:
            input_columns = st.columns(5)  # Create 5 columns for input fields
        
            user_input = {}
            for field, choices in self.input_sections.items():
                if choices != []:
                    # Display each selectbox in a separate column
                    user_input[field] = input_columns[len(user_input) % 5].selectbox(field, choices)
                elif field == "SeniorCitizen":
                    input_dummy = input_columns[len(user_input) % 5].selectbox(field, ["Yes", "No"])
                    user_input[field] = 1 if input_dummy == "Yes" else 0
                else:
                    if field == "tenure":
                        user_input[field] = input_columns[len(user_input) % 5].number_input(field, value=0, step=1)
                    else:
                        user_input[field] = input_columns[len(user_input) % 5].number_input(field, value=0.0)
                        
                        
        input_df = pd.DataFrame(user_input, index=[0])
        with result_columns[1]:
            if st.button("Predict"):
                predicted_df = self.model.predict_churn(input_df)

                # Display predictions in the result column
                self.display_prediction(predicted_df["predicted_churn"].iloc[0],
                                        predicted_df["prediction_confidence"].iloc[0])
                
        st.sidebar.header("Batch Processing")
        st.sidebar.caption("The file must contain the same columns shown in the single prediction screen.")
        batch_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        print(batch_file)
        if batch_file is not None:
            batch_input_df = pd.read_csv(batch_file)
            
            if st.button("Batch Predict"):
                batch_predictions_df = self.model.predict_churn(batch_input_df)
                st.dataframe(batch_predictions_df)
            

if __name__ == "__main__":
    app = ChurnApp()
    app.run_app()
