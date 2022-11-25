from sklearn.base import BaseEstimator
from src.artifact_manager import ArtifactManager

class Trainer():

    def __init__(self,*args,**kwargs) -> None:
       
        self._model:BaseEstimator=None

    @property
    def model(self)->BaseEstimator:
        return self._model

    @model.setter
    def model(self,obj:BaseEstimator):
        self._model=obj

    @model.deleter
    def model(self)->None:
        del self._model
    
    def fit(self,x,y)->BaseEstimator:
        return self.model.fit(x,y)



