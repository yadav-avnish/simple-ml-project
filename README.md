Setup this project

Create virtural environment
```
conda create -p venv python==3.8 -y
```
Activate virtual environment
```
conda activate venv
```
Install dependencies
```
pip install -r requirements.txt
```
Launch application to perform prediction
```
uvicorn app:app --host=0.0.0.0 --reload
```


