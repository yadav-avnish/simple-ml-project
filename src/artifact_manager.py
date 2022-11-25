
from datetime import datetime
from sklearn.compose import ColumnTransformer
import os
from sklearn.base import BaseEstimator
import dill
from typing import List,Optional
from dataclasses import dataclass
import shutil
@dataclass
class TransformerAndModel:
    transformer:ColumnTransformer
    model:object

class ArtifactManager:

    def __init__(self,artifact_dir:str="artifact",*args,**kwargs) -> None:
        try:
            self.artifact_dir:str=os.path.join(artifact_dir)
            os.makedirs(self.artifact_dir,exist_ok=True)            
            self.__transformer_file_path=os.path.join("data_transform","data_transform.pkl")
            self.__model_file_path=os.path.join("model","model.pkl")
            self._loaded_transformer_path = None
            self._loaded_model_path = None
            self._loaded_transformer=None
            self._loaded_model = None
            
        except Exception as e:
            raise e

    @property
    def list_artifact_num(self):
        return os.listdir(self.artifact_dir)
        
    
    def _get_dir(self,to_save:bool=True)->str:
        try:
            numeric_dir:List[int] = list(map(int,os.listdir(self.artifact_dir)))
            if len(numeric_dir)==0:
                return os.path.join(self.artifact_dir,f"0")
            else:
                previous_latest_dir = f"{max(numeric_dir)}"
            
            if to_save:
                if len(os.listdir(os.path.join(self.artifact_dir,previous_latest_dir)))!=0:
                    previous_latest_dir = f"{max(numeric_dir)+1}"

            return os.path.join(self.artifact_dir,previous_latest_dir)
        except Exception as e:
            raise e

    def save_transformer(self,transformer:ColumnTransformer):
        try:
            self.artifact_dir = self._get_dir()
            transformer_file_path = os.path.join(self.artifact_dir,self.__transformer_file_path)
            os.makedirs(os.path.dirname(transformer_file_path),exist_ok=True)
            with open(transformer_file_path,"wb") as transformer_writer:
                dill.dump(transformer,transformer_writer)
            print(f"Transformer object is saved at {transformer_file_path}")
        except Exception as e:
            raise e
        
    def save_model(self,model:object):
        try:
            model_file_path = os.path.join(self.artifact_dir,self.__model_file_path)
            if os.path.basename(model_file_path)==model_file_path:
                raise Exception("First save you transformer object")
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            with open(model_file_path,"wb") as model_writer:
                dill.dump(model,model_writer)
            print(f"Model object is saved at {model_file_path}")
        except Exception as e:
            raise e

    def load_transformer(self,artifact_num:Optional[int]=None)->ColumnTransformer:
        try:
            if artifact_num is None:
                transformer_file_path = os.path.join(self._get_dir(to_save=False),self.__transformer_file_path)
            else:
                transformer_file_path = os.path.join(self.artifact_dir,f"{artifact_num}",self.__transformer_file_path)
            if not os.path.exists(transformer_file_path):
                raise Exception(f"Transformer is not available at: {transformer_file_path}")
            
            if self._loaded_transformer_path!=transformer_file_path:
                with open(transformer_file_path,"rb") as transformer_reader:
                    transformer =  dill.load(transformer_reader)
                self._loaded_transformer_path=transformer_file_path
                self._loaded_transformer=transformer
                return transformer
            else:
                return self._loaded_transformer
        except Exception as e:
            raise e

    def load_model(self,artifact_num:Optional[int]=None):
        try:
            model=None
            if artifact_num is None:
                model_file_path = os.path.join(self._get_dir(to_save=False),self.__model_file_path)
            else:
                model_file_path = os.path.join(self.artifact_dir,f"{artifact_num}",self.__model_file_path)
            if not os.path.exists(model_file_path):
                raise Exception(f"Model is not available at: {model_file_path}")
            if self._loaded_model_path!=model_file_path:
                    
                with open(model_file_path,"rb") as model_reader:
                    model =  dill.load(model_reader)
                
                self._loaded_model_path=model_file_path
                self._loaded_model=model
                return model
            else:
                return self._loaded_model
        except Exception as e:
            raise e
        
    def load_transform_n_model(self,artifact_num:Optional[int]=None)->TransformerAndModel:
        try:
            transformer:ColumnTransformer= self.load_transformer(artifact_num=artifact_num)
            model:object= self.load_model(artifact_num=artifact_num)
            return TransformerAndModel(transformer=transformer,model=model)
        except Exception as e:
            raise e
    def save_transformer_n_model(self,transformer_n_model:TransformerAndModel,):
        try:
            self.save_transformer(transformer=transformer_n_model.transformer,)
            self.save_model(model=transformer_n_model.model,)
        except Exception as e:
            raise e
        


        
   