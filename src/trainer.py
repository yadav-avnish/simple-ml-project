from sklearn.base import BaseEstimator
from src.artifact_manager import ArtifactManager

class Trainer():

    def __init__(self,*args,**kwargs) -> None:
       
        self._model:object=None

    @property
    def model(self)->object:
        return self._model

    @model.setter
    def model(self,obj:object):
        self._model=obj

    @model.deleter
    def model(self)->None:
        del self._model
    
    def fit(self,x,y)->object:
        return self.model.fit(x,y)



