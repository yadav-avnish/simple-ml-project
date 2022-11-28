## Insurance Premium Prediction 

## Docker File
```Dockerfile
# Use python 3.9.15 version
FROM python:3.9.15-slim-bullseye

# Copy content to app dir
COPY . app/

# Change dir to app
WORKDIR app/

# Install Requirements 
RUN pip install -r requirements.txt 

# Train the model
RUN python train.py

# Run Prediction
CMD ["python","app.py"]

```
Build docker image
```bash
docker build -t ml-app .
```
Create container and run
```bash
docker run -p 8000:8000 --name app ml-app    
```
Remove stopped containers 
```
docker system prune -f
```

### Local Setup
Create virtural environment
```
conda create -p ./env python==3.9 -y
```
Activate virtual environment
```
conda activate ./env
```
Install dependencies
```
pip install -r requirements.txt
```
Launch application to perform prediction
```
uvicorn app:app --host=0.0.0.0 --reload
```


