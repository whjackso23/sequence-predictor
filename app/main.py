import os
from typing import List, Optional
import torch
from datetime import date
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
from utils import (
    download_artifacts,
    untar_artifacts,
    load_artifacts,
    convert_json_to_map,
    map_to_tensor,
    get_classifier,
    Settings,
)
import logging
from filelock import FileLock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%F %T"
)
logger = logging.getLogger(__name__)

# Pydantic type checking for events json passed to /predict
class WheelsetReading(BaseModel):
    train_entrance_ts: date = Field(..., example="2018-06-30")
    wheel_dyn_l: float = Field(..., example=7.89)
    wheel_dyn_r: float = Field(..., example=3.67)


class Axle(BaseModel):
    wheelsets: List[WheelsetReading]
    axle_seq_nbr: int = Field(..., example=1)
    total_miles: Optional[float] = Field(0, example=15.859)
    association_date: Optional[str] = Field(None, example="2018-06-30")


class Mileage(BaseModel):
    event_date: Optional[str] = Field(None, example="2018-06-30")
    daily_miles: Optional[float] = Field(None, example=15.859)


class PredictRequest(BaseModel):
    axle: List[Axle]
    mileage: List[Mileage]
    equipment_initial: str = Field(..., example="GATX")
    equipment_number: str = Field(..., example="0000123456")
    equipment_type_code: str = Field(..., example="A100")


class CRUL(BaseModel):
    axle_seq_nbr: int = Field(..., example=1)
    probability: float = Field(..., example=0.8595)


class CRULResponse(BaseModel):
    predictions: List[CRUL]


class Alive(BaseModel):
    status: bool = Field(..., example=True)


# def initialize(env):
#     config = EnvYAML("../config/app_config.yml")
#     global s3_config
#     s3_config = config["s3"][env]
#
#
# parameter_name = f"/app/env"
# try:
#     ssm_client = boto3.client("ssm", region_name="us-east-1")
#     env = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)[
#         "Parameter"
#     ]["Value"]
# except Exception as e:
#     raise e
# initialize(env)
# logger.info(s3_config["artifact_bucket"])

# currently need ot hardcode artifact path until Sagemaker permissions set to pull from model registry
settings = Settings(
    artifact_bucket=os.getenv("ARTIFACT_BUCKET"),
    version=os.getenv("VERSION"),
    artifact_path=os.getenv("ARTIFACT_PATH"),
    artifact_name=os.getenv("ARTIFACT_NAME"),
    model_name=os.getenv("MODEL_NAME"),
    vocab_name=os.getenv("VOCAB_NAME"),
)
# settings = Settings()

models = {}

app = FastAPI(
    title="Component Remaining Useful Life Prediction Service",
    description="Component Remaining Useful Life prediction service that takes a sequence of mileage, wheelsets, and equipment types then returns something probably",
    version=settings.version,
    docs_url="/crul/predict-service/docs",
    redoc_url=None,
)

internal = {500: {"description": "Internal Application Error"}}

no_events = {400: {"description": "Error: Bad Request"}}


@app.on_event("startup")
def on_startup():
    # pull all artifacts local
    with FileLock(f"{settings.artifact_name}.lock"):
        try:
            vocab, classifier = load_artifacts(settings.vocab_name, settings.model_name)
        except FileNotFoundError:
            download_artifacts(
                settings.artifact_bucket, settings.artifact_path, settings.artifact_name
            )
            logger.info(f"downloaded {settings.artifact_path}")
            untar_artifacts(settings.artifact_name)
            vocab, classifier = load_artifacts(settings.vocab_name, settings.model_name)
            logger.info(f"loaded model to worker")
    models["ALL"] = (classifier, vocab)


@app.get(
    "/crul/predict-service/status",
    summary="Service Status",
    response_model=Alive,
    response_description="Service Up",
)
def read_root():
    """
    Check to see if to see if service is up.
    """
    return {"status": True}


# @app.get(
#     "/crul/predict-service/artifacts",
#     summary="Get the deployed model artifact directory names from S3",
#     response_model=ArtifactList,
#     response_description="List of Available Models",
#     responses={**no_od},
# )
# def artifacts():
#     """
#     Get a list of available artifact directories
#     """
#     dirlist = [
#         item
#         for item in os.listdir(settings.artifact_path)
#         if os.path.isdir(os.path.join(settings.artifact_path, item))
#     ]
#     logger.info(dirlist)
#     if len(dirlist) == 0:
#         raise HTTPException(status_code=204, detail="No Model Artifacts")
#     return {"experiments": dirlist}


@app.post(
    "/crul/predict-service/predict",
    summary="Predict remaining useful life of component(s) for a provided equipment.",
    response_description="Successful prediction",
    responses={**no_events, **internal},
)
async def predict(prediction: PredictRequest):
    """
    Predict remaining useful life of wheelsets for provided equipment and return something.
    - **equipment_initial**: Stencilled initial of equipment
    - **equipment_number**: Stencilled number of equipment
    - **equipment_type_code**: Umler UMET for provided equipment
    - **mileage**: sequence of daily miles for 90 days, sorted in ascending order by date with the following details
        - event_date: day of mileage accruing
        - daily_miles: Miles accrued on day
    - **axle**: Information about the specific wheelset axle in question
        - axle_seq_nbr: Sequence number of the axle on the car
        - cumulative_miles: Miles accrued since installation of wheelset
        - wheelset_readings: A list of information about wheelset readings including:
            - train_entrance_ts: datetime of most recent reading
            - wheel_dynamic_left: difference in wheel vertical peak and nominal (left side)
            - wheel_dynamic_right: difference in wheel vertical peak and nominal (right side)
    """
    # model = get_artifacts(prediction, models)
    model, vocab = get_classifier(models)
    model.eval()
    json_dict = jsonable_encoder(prediction)
    logger.info(f"PAYLOAD {json_dict}")
    axle_dict = convert_json_to_map(json_dict, vocab)
    logger.info(f"PROCESSED PAYLOAD {axle_dict}")
    out_response = []
    for axle_seq in range(1, axle_dict["axle_count"] + 1):
        try:
            history = len(axle_dict[axle_seq]["reading_event_date_list"])
        except KeyError:
            continue
        if history < 5:
            raise HTTPException(status_code=204, detail="Insufficient WILD information")
        else:
            try:
                X1, X2 = map_to_tensor(axle_dict, axle_seq)
            except KeyError:
                continue
            output = model(X1, X2)
            prob = torch.sigmoid(output)
            tmp = {"axle_seq_nbr": axle_seq, "probability": prob.item()}
        out_response.append(tmp)
    logger.info(f"RESULT {out_response}")
    return out_response
