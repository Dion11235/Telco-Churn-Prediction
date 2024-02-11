# Telco-Churn-Prediction
This is a churn classification project on a fictitious company Telco prepared by IBM.
Dataset link : https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

In this repository I have tried to build a well structured ML project, maintaining the directory hirarchy as well as model tracking for different experiments.
At the end the model is deployed in Streamlit. You can try it out here: [Live demo link](https://telco-churn-prediction.streamlit.app/)

A quick sneak peek to the app :
![app sneak peek](https://github.com/Dion11235/Telco-Churn-Prediction/blob/main/plots/app_peek.png?raw=True)

## How to reproduce the results ?
 - step 1 : open a terminal in your local project directory.
 - step 2 : run `git clone https://github.com/Dion11235/Telco-Churn-Prediction/`
 - step 3 : run `conda create -p churnenv python=3.9` to create the environment
 - step 4 : run `cd Telco-Churn-Prediction` to move into the cloned directory.
 - step 5 : run `pip install -r requirements.txt` to install all the dependencies.
 - step 6 : run `python src/pipeline/train_pipeline.py` to run the experiments. Dont forget to delete the folder *mlflow_experiments_log* before running this, otherwise there will be duplicate experiments.
 - step 7 : run `mlflow ui --backend-store-uri=mlflow_experiments_log` to check out the awesome mlflow dashboard!

## Results :
Achieved F1 score 0.865 and AUC : 0.86, ROC curves for different experiments look like this -
![ROC Curves for different experiments](https://github.com/Dion11235/Telco-Churn-Prediction/blob/main/plots/roc_curve_all_models.png?raw=True)


If you like the work, give a star :star2:. Thanks in advance! 😄
