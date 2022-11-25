
from sklearn.pipeline import Pipeline
import logging
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
import numpy as np
from typing import Optional
from src.artifact_manager import ArtifactManager
# Define input feature names
ORDINAL_COLUMNS = ['sex','smoker']
ONE_HOT_COLUMNS = ['region']
NUMERICAL_COLUMNS = ["age","bmi","children"]
TARGET_COLUMNS = "expenses"
# Define dataset Type
@dataclass
class Dataset:
    input:np.ndarray
    target:np.ndarray


    def __getitem__(self,idx):
        return self.input[idx,:],self.target[idx]

    @property
    def shape(self):
        shape_info = {"input":self.input.shape,"target":self.target.shape}
        return shape_info

    
#Define output of Transform Dataset
@dataclass
class TransformDatasetArtifact:
    train:Dataset
    val:Dataset
    test:Dataset


class Transformer():

    def __init__(self,dataset_path:str,random_state:int=74,*args,**kwargs) -> None:
        """
        dataset_path:str Datset file location
        """
        try:
          
            self.__dataset_path:str = dataset_path
            self.random_state:int = random_state
            self.__transformer_obj:Optional[ColumnTransformer] = None


        except Exception as e:
            raise e
    @property
    def transformer_obj(self)->Optional[ColumnTransformer]:
        try:
            if self.__transformer_obj is None:
                print("Transformer obj is not yet trained")
            return self.__transformer_obj
        except Exception as e:
            raise e
    
    @staticmethod
    def get_transformer()->ColumnTransformer:
        try:
            
            #numerical transformation
            numerical_pipleine:Pipeline = Pipeline(
                steps=[
                    ("standard_scaler",StandardScaler())
                ]
            )
            #categorical transformation
            one_hot_pipleine:Pipeline = Pipeline(
                steps=[
                    ("one_hot",OneHotEncoder()),
                
                ]
            )
            ordinal_pipeline:Pipeline = Pipeline(
                steps=[
                    ("one_hot",OrdinalEncoder()),
                
                ]
            )
            
            #combining all transformation
            transformer:ColumnTransformer = ColumnTransformer(transformers=[
                ('one_hot',one_hot_pipleine,ONE_HOT_COLUMNS),
                ('ordinal',ordinal_pipeline,ORDINAL_COLUMNS),
                ('numerical',numerical_pipleine,NUMERICAL_COLUMNS),
            ])
            return transformer
        except Exception as e:
            raise e
        
    def __call__(self)->TransformDatasetArtifact:
        try:
            transformer:ColumnTransformer = self.get_transformer()
            
            logging.debug(f"Transformer object: {transformer}")
            #read dataframe
            df:pd.DataFrame = pd.read_csv(filepath_or_buffer=self.__dataset_path)
            logging.debug(f"Dataset shape: {df.shape}")
            x:np.ndarray = transformer.fit_transform(df)
            self.__transformer_obj=transformer
            logging.debug(f"Input feature shape: {x.shape}")
            y:np.ndarray = df[TARGET_COLUMNS].to_numpy()
            logging.debug(f"Target feature shape: {y.shape}")
            x, x_val, y, y_val = train_test_split(x, y, test_size=0.10, random_state=self.random_state)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=self.random_state)
            
            train_dataset:Dataset = Dataset(input=x_train,target=y_train)
            logging.debug(f"Train dataset shape: {train_dataset.shape}")
            
            val_dataset:Dataset = Dataset(input=x_val,target=y_val)
            logging.debug(f"Validation dataset shape: {val_dataset.shape}")
            test_datatset:Dataset = Dataset(input=x_test,target=y_test)
            logging.debug(f"Testing dataset shape: {test_datatset.shape}")

            return TransformDatasetArtifact(train=train_dataset,val=val_dataset,test=test_datatset)
        except Exception as e:
            raise e


