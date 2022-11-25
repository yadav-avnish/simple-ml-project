from fastapi import FastAPI
from dataclasses import dataclass
import pandas as pd
import uvicorn
from src.artifact_manager import ArtifactManager
from pydantic import BaseModel
from typing import Optional
artifact_manager = ArtifactManager()

SEX_FEATURE_ACCEPTABLE_DATA=["male","female"]
SMOKER_FEATURE_ACCEPTABLE_DATA = ["yes","no",]
REGION_FEATURE_ACCEPTABLE_DATA = ["northwest","northeast","southeast","southwest"]

class CustomerDetail(BaseModel):
    """
    age: 19
    sex: Male

    """
    age:int=19
    sex:str="female"
    bmi:float=21.04
    children:int=0
    smoker:str="yes"
    region:str="southwest"
    expenses:Optional[float]=None

    def validate_data(self):
        error_message = ""
        if self.sex not in SEX_FEATURE_ACCEPTABLE_DATA:
            error_message+=f"Sex feature: {self.sex} is not valid value acceptable values are {SEX_FEATURE_ACCEPTABLE_DATA} "

        if self.smoker  not in SMOKER_FEATURE_ACCEPTABLE_DATA:
            error_message+=f"Smoker feature: {self.smoker} is not valid value acceptable values are {SEX_FEATURE_ACCEPTABLE_DATA} "
        
        if self.region not in REGION_FEATURE_ACCEPTABLE_DATA:
            error_message+=f"Region feature {self.region} is not valid value acceptable values are {REGION_FEATURE_ACCEPTABLE_DATA} "
        if len(error_message)>0:
            raise Exception(error_message)

    def to_df(self):
        print(self.__dict__)
        rows = [list(self.__dict__.values())]
        print(rows)
        columns = list(self.__dict__.keys())
        print(columns)
        df =  pd.DataFrame(data=rows)
        df.columns=columns
        print(df.values)
        return df

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict",)
async def predict_premium(customer_detail:CustomerDetail):
    try:
        #validate data
        customer_detail.validate_data()
        df = customer_detail.to_df()
        transformer_n_model = artifact_manager.load_transform_n_model()
        transform_arr = transformer_n_model.transformer.transform(df)
        print(transform_arr)
        prediction = transformer_n_model.model.predict(transform_arr)
        customer_detail.expenses=prediction[0]
        return customer_detail
    except Exception as e:
        return {"error":str(e)}


if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
