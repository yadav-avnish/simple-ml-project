from src.preprocessor import Transformer
from src.trainer import Trainer
from src.artifact_manager import ArtifactManager,TransformerAndModel
import argparse
import os
import pprint
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.evaluate import is_model_acceptable
from sklearn.metrics import r2_score

logging.getLogger().setLevel(logging.DEBUG)
DATASET_FILE_PATH=os.path.join("dataset","insurance.csv")

def train(dataset_file_path:str):
    artifact_manager = ArtifactManager()
    transformer = Transformer(dataset_path=dataset_file_path)
    dataset = transformer()
    trainer = Trainer()
    trainer.model = RandomForestRegressor()
    model = trainer.fit(x=dataset.train.input,y=dataset.train.target)
    transformer_n_model = TransformerAndModel(transformer=transformer.transformer_obj,model=model)
    
    print(f">"*20,"Model Score",f"<"*20)
    prediction = model.predict(dataset.train.input)
    print("Training score",r2_score(dataset.train.target,prediction))
    prediction = model.predict(dataset.val.input)
    print("Validation score",r2_score(dataset.val.target,prediction))
    prediction = model.predict(dataset.test.input)
    print("Testing score",r2_score(dataset.test.target,prediction))

    if is_model_acceptable(data_transformer=transformer.transformer_obj,model=model,dataset_file_path=dataset_file_path):
        print("Trained model is better hence accepting the trained model")
        artifact_manager.save_transformer_n_model(transformer_n_model=transformer_n_model)
        return None
    print("Trained model is not better hence trained model is rejected.")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file_path", default=DATASET_FILE_PATH, type=str, help="If provided true training will be done else not")
    args = parser.parse_args()
    print(args.dataset_file_path)
    train(dataset_file_path=args.dataset_file_path)
