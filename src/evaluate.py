

from .preprocessor import TransformDatasetArtifact
from sklearn.compose import ColumnTransformer
from .artifact_manager import ArtifactManager
from dataclasses import dataclass
from sklearn.metrics import r2_score
import pandas as pd
from src.preprocessor import TARGET_COLUMNS
artifact_manager = ArtifactManager()






def get_model_score(data_transformer:ColumnTransformer,model,df:pd.DataFrame)->float:
    try:
        input_arr  = data_transformer.transform(df)
        pred = model.predict(input_arr)
        return r2_score(df[TARGET_COLUMNS],pred)
    except Exception as e:
        raise e



def is_model_acceptable(data_transformer,model,dataset_file_path:str)->bool:
    try:
        if len(artifact_manager.list_artifact_num)==0:
            print("Model not available for comparision hence accepting trained model")
            return True
        df = pd.read_csv(dataset_file_path)
        transform_n_model = artifact_manager.load_transform_n_model()
        
        previous_model_score = get_model_score(data_transformer=transform_n_model.transformer,model=transform_n_model.model,df=df)
        model_score=  get_model_score(data_transformer= data_transformer,model=model,df=df)
        print(f"Improved accuracy: {model_score-previous_model_score}")
        if previous_model_score<model_score:
            
            return True
        else:
            return False
    except Exception as e:
        raise e
