FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app

RUN chmod +x scripts/three_models_bestpicture.sh && pip install -r requirements.txt && \
    apt-get update && apt-get install nano 
    
ENTRYPOINT ["scripts/three_models_bestpicture.sh"]