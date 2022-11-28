FROM python:3.9.15-slim-bullseye

COPY . app/

WORKDIR app/

RUN pip install -r requirements.txt 

RUN python train.py

CMD ["python","app.py"]