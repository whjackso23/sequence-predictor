FROM tiangolo/uvicorn-gunicorn-fastapi
RUN pip3 install filelock pandas envyaml scipy boto3 pytorch-lightning==1.4.5
COPY ./app /app
COPY ./app.env /app
COPY ./config /config

# test service startup ( logging to std out )
# docker run -it --env-file=app.env -p 80:80 db68a992a27d

# Run as detached service
#docker run -d --name crul-predict \
#    --env-file=app.env \
#    --log-driver=awslogs \
#    --log-opt awslogs-region=us-east-1 \
#    --log-opt awslogs-group=crul-wild-classifier \
#    -p 80:80 crul-predict:latest
