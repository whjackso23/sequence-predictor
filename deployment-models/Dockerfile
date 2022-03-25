FROM python:3.6-slim
WORKDIR /deploy-models
COPY . /deploy-models
RUN pip install --no-cache-dir -r requirements.txt

# example run
# docker run --name deploy-models --env-file=deploy.env deploy-models:latest python3 deploy_models.py --experiment_names deep_dodo_457210000_344192000 cool_aphid_327488000_076057000