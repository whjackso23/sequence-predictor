import mlflow
import re
import boto3
import os
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_names", nargs="+", type=str)
args = parser.parse_args()


class DeployMLFlow:
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("ML_FLOW_TRACKING"))
    s3 = boto3.resource("s3")

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.od = "_" + re.match(r".+_(\d+_\d+)", self.experiment_name).group(1)
        self.mlflow_bucket = os.getenv("MLFLOW_BUCKET")
        print("mlflow_bucket: " + self.mlflow_bucket)
        self.mlflow_bucket_obj = self.set_bucket_obj(self.mlflow_bucket)
        self.deploy_bucket = os.getenv("DEPLOY_BUCKET")
        print("deploy_bucket: " + self.deploy_bucket)
        self.deploy_bucket_obj = self.set_bucket_obj(self.deploy_bucket)
        self.deploy_folder = os.getenv("DEPLOY_FOLDER")
        print("deploy_folder: " + self.deploy_folder)
        self.deploy_prefix = f"{self.deploy_folder}/{self.od}/"
        # all keys for experiment
        self.file_objects = self.get_file_objects()

    def get_file_objects(self):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        run = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string='tags.mlflow.runName="best_model"',
        )
        run_id = run.run_id.values[0]
        with mlflow.start_run(run_id=run_id):
            uri = mlflow.get_artifact_uri()
        search_prefix = uri.split(f"s3://{self.mlflow_bucket}/")[1]
        return self.mlflow_bucket_obj.objects.filter(Prefix=search_prefix)

    def set_bucket_obj(self, s3_bucket):
        return self.s3.Bucket(s3_bucket)

    def get_best_model_key(self):
        # The best model is in the lowest numbered checkpoint directory.
        #  ( checkpoint nbr, s3 file object )
        checkpoints = [
            (int(re.match(r".+checkpoint_(\d+).+", x.key).group(1)), x)
            for x in self.file_objects
            if "checkpoint_" in x.key and "model.pth" in x.key
        ]
        checkpoints = sorted(checkpoints, key=lambda x: x[0])
        return checkpoints[0][1].key

    def get_best_params_key(self):
        params = [x for x in self.file_objects if "params.json" in x.key]
        return params[0].key

    def get_best_vocab_key(self):
        vocab = [x for x in self.file_objects if "vocab.pkl" in x.key]
        return vocab[0].key

    def clear_target_folder(self):
        self.deploy_bucket_obj.objects.filter(Prefix=self.deploy_prefix).delete()

    def copy_object(self, source_key, object_name):
        copy_source = {"Bucket": self.mlflow_bucket, "Key": source_key}
        print(copy_source)
        self.deploy_bucket_obj.copy(copy_source, self.deploy_prefix + object_name)

    def copy_objects(self):
        artifact.clear_target_folder()
        self.copy_object(self.get_best_model_key(), "best_model.pth")
        self.copy_object(self.get_best_params_key(), "params.json")
        self.copy_object(self.get_best_vocab_key(), "vocabs.pkl")


_experiment_name = args.experiment_names

for _x in _experiment_name:
    artifact = DeployMLFlow(_x)
    artifact.copy_objects()
