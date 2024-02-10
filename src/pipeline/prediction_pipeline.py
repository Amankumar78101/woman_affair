import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 rate_marriage:float,
                 age:float,
                 yrs_married:float,
                 children:float,
                 religious:float,
                 educ:float,
                 occupation:float,
                 occupation_husb:float):
        
        self.rate_marriage=rate_marriage
        self.age=age
        self.yrs_married=yrs_married
        self.children=children
        self.religious=religious
        self.educ=educ
        self.occupation = occupation
        self.occupation_husb = occupation_husb

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'rate_marriage':[self.rate_marriage],
                'age':[self.age],
                'yrs_married':[self.yrs_married],
                'children':[self.children],
                'religious':[self.religious],
                'educ':[self.educ],
                'occupation':[self.occupation],
                'occupation_husb':[self.occupation_husb]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
