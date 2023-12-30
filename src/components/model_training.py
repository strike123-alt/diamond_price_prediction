import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.logger import logging 
from src.utils import save_object
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
import sys
import os 
from src.utils import evaluate_model

@dataclass 
class ModelTrainerConfig: 
    trained_model_file_path:str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self, train_array, test_array): 
        try: 
            logging.info('Splitting Dependent and Independent variables from train and test')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet()

            }
            evaluate_model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(evaluate_model_report)
            print(f'\n{"="*75}\n')
            logging.info(f'Model Report : {evaluate_model_report}')
            best_model_score = max(sorted(evaluate_model_report.values()))
            best_model_name = list(evaluate_model_report.keys())[
                list(evaluate_model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(f'Best Model Found, Model Name : {best_model_name}, R2 Score: {best_model_score}')
            print(f'\n{"=" * 75}\n')
            logging.info('Best Model Found, Model Name: {best_model_name}, r2 Score: {best_model_score}')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e: 
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)