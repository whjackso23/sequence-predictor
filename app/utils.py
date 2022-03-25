import pandas as pd
from scipy import signal
import numpy as np
import torch
from model import TCN
from pydantic import BaseModel
from datetime import datetime, date, timedelta
import logging
import tarfile
import boto3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%F %T"
)
logger = logging.getLogger(__name__)


class Settings(BaseModel):
    artifact_bucket: str
    version: str
    artifact_path: str
    artifact_name: str
    model_name: str
    vocab_name: str


def convert_json_to_map(json_dict, vocab):
    axle_dict = {}
    axle_count = 0
    equipment_type_code = json_dict["equipment_type_code"]
    umet = vocab["umet"].get(equipment_type_code, vocab["umet"]["<unk>"])
    axle_dict["umet"] = [umet]
    for axle in json_dict["axle"]:
        axle_seq_nbr = axle["axle_seq_nbr"]
        if axle["association_date"]:
            association_date = datetime.strptime(
                axle["association_date"][:10], "%Y-%m-%d"
            ).date()
            logger.info(f"Association found: {association_date}")
        else:
            association_date = date.today() - timedelta(days=365)
            logger.info(f"No association found using {association_date}")
        axle_count = max(axle_seq_nbr, axle_count)
        axle_dict[axle_seq_nbr] = {}
        axle_dict[axle_seq_nbr]["left_read_list"] = []
        axle_dict[axle_seq_nbr]["right_read_list"] = []
        axle_dict[axle_seq_nbr]["reading_event_date_list"] = []
        axle_dict[axle_seq_nbr]["association_date"] = association_date
        for reading in axle["wheelsets"]:
            entrance_ts = datetime.strptime(
                reading["train_entrance_ts"][:10], "%Y-%m-%d"
            ).date()
            if entrance_ts < association_date:
                axle_dict[axle_seq_nbr]["left_read_list"].append(0)
                axle_dict[axle_seq_nbr]["right_read_list"].append(0)
                axle_dict[axle_seq_nbr]["reading_event_date_list"].append(entrance_ts)
            else:
                axle_dict[axle_seq_nbr]["left_read_list"].append(reading["wheel_dyn_l"])
                axle_dict[axle_seq_nbr]["right_read_list"].append(
                    reading["wheel_dyn_r"]
                )
                axle_dict[axle_seq_nbr]["reading_event_date_list"].append(entrance_ts)
    axle_dict["axle_count"] = axle_count
    return axle_dict


def get_wild_seq(date_range, wild_dates, wild):
    """
    get wild seq for left wheel
    :param df: data frame with array of wild readings
    :return: data frame with processed array of wild readings
    """
    ## med filter applied asssuming reads are in in ascending order by date (do we need check)
    dyn = pd.Series(signal.medfilt([float(x) for x in wild], 5), index=wild_dates)
    dyn.index = pd.to_datetime(dyn.index)
    dyn = dyn.resample("D").max().reindex(date_range)
    dyn = list(
        dyn.interpolate(
            method="spline", order=3, limit_direction="forward", limit_area="inside"
        )
        .ffill()
        .fillna(0)
    )[2:]
    return dyn


def map_to_tensor(map, axle_seq, seq_len=90):
    date_range = pd.date_range(
        (pd.to_datetime("today") - pd.to_timedelta(seq_len + 1, unit="d")),
        pd.to_datetime("today"),
        freq="d",
    ).normalize()
    l_dyn_vert = get_wild_seq(
        date_range,
        map[axle_seq]["reading_event_date_list"],
        map[axle_seq]["left_read_list"],
    )
    r_dyn_vert = get_wild_seq(
        date_range,
        map[axle_seq]["reading_event_date_list"],
        map[axle_seq]["right_read_list"],
    )
    X1 = np.stack([l_dyn_vert, r_dyn_vert], axis=0)
    X1 = np.expand_dims(X1, axis=0)
    X2 = np.array([map["umet"] * seq_len])
    X2 = np.expand_dims(X2, axis=2)
    return torch.Tensor(X1.astype(float)), torch.Tensor(X2.astype(float))


def download_artifacts(bucket, model_path, artifact_name):
    """
    Function to download artifacts for model you want to evaluate

    """
    s3 = boto3.client("s3")
    s3.download_file(bucket, model_path, artifact_name)


def untar_artifacts(artifact_name):
    """
    Untar downlaoded file
    """
    file = tarfile.open(artifact_name)
    logger.info(f"extracting: {file.getnames()}")
    file.extractall(".")
    file.close()


def load_artifacts(vocab, model):
    """
    Load vocab and model object
    """
    vocab = torch.load(f"{vocab}")
    # ideally would pull from model registry but permissions not set up yet
    classifier = TCN(
        vocab=vocab,
        num_size=2,
        output_size=1,
        num_channels=[8 * 5],
        kernel_size=28,
        dropout=0,
        learning_rate=0,
        model_type="classifier",
    )
    state_dict = torch.load(f"{model}", map_location=torch.device("cpu"))
    classifier.load_state_dict(state_dict)
    return vocab, classifier


def get_classifier(models, equip="ALL"):
    return models[equip]
