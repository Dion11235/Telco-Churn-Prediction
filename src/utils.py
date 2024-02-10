import os
import pickle
import json
import matplotlib.pyplot as plt
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        return obj

    except Exception as e:
        raise CustomException(e)
    

def save_json(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e)
    

def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            obj = json.load(f)
        return obj
    
    except Exception as e:
        raise CustomException(e)
            
    

def generate_roc_curves(experiment_names, all_fpr, all_tpr, all_auc, plot_file_name):
    os.makedirs(os.path.dirname(plot_file_name), exist_ok=True)
    plt.figure(figsize=(8, 8))

    for i, exp_name in enumerate(experiment_names):
        plt.plot(all_fpr[i], all_tpr[i], label=f'Model {exp_name} (AUC = {all_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - All Models')
    # plt.legend(loc="lower right")
    plt.legend(loc="upper right", bbox_to_anchor=(1.65, 1), borderaxespad=0, fontsize='small')

    plt.savefig(plot_file_name, bbox_inches='tight')
    plt.close()