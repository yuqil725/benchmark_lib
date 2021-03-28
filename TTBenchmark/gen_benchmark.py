import collections
import json
import os

import tensorflow as tf
from model_data_util.create_tt_data.model_data_convert import convertModelToRawData

from TTBenchmark.check_environment import check_env_info
from TTBenchmark.constant import GDRIVE_PATH


class BenchmarkDataMini():
    def __init__(self,
                 raw_model: tf.keras.Model,
                 actual_tt_mean: float,
                 actual_tt_median: float,
                 actual_tt_std: float,
                 batch_size: int,
                 validation_split: float,
                 verbose=False
                 ):
        self.model_info: dict = {
            "raw_model": raw_model
        }
        self.actual_tt: dict = {
            "mean": actual_tt_mean,
            "median": actual_tt_median,
            "std": actual_tt_std
        }
        self.fit_kwargs: dict = {
            "batch_size": batch_size,
            "validation_split": validation_split,
            "verbose": verbose
        }


def _ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def _write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def save_benchmark(
        benchmarks_mini: list,
        columns: list,
        model_type: str,
        gdrive_path=GDRIVE_PATH,
        replace=False
):
    """
    save benchmarks into json structure:
        model_name1:{
            model_df: df,
            actual_tt: {mean: , median: , std:}
            fit_info: {batch_size: , optimizer: , validation_split: , verbose: }
        }
    :param benchmarks_mini: list of benchmark_mini
    :param columns: 
    :param model_type: 
    :param gdrive_path: 
    :return: 
    """
    env_fname = "_".join(list(check_env_info().values()))
    env_path = os.path.join(gdrive_path, env_fname)
    actual_tt_json_path = os.path.join(env_path, model_type, "trained_tt.json")
    _ensure_dir(actual_tt_json_path)
    model_index = None
    if not os.path.exists(actual_tt_json_path) or replace == True:
        actual_tt_json = collections.defaultdict(dict)
        _write_json(actual_tt_json, actual_tt_json_path)
        model_index = 0

    with open(actual_tt_json_path) as f:
        for bmdatamini in benchmarks_mini:
            model = bmdatamini.model_info["raw_model"]
            actual_tt = bmdatamini.actual_tt
            fit_kwargs = bmdatamini.fit_kwargs
            num_data = bmdatamini.training_size

            actual_tt_json = json.load(f)
            if model_index is None:
                model_index = int(max(actual_tt_json["model_name"]).split("_")[-1]) + 1

            model_name = f"{model_type}_{model_index}"

            item = actual_tt_json[model_name]
            item["model_df"] = convertModelToRawData(model, columns, num_data)
            item["actual_tt"] = actual_tt
            item["fit_kwargs"] = fit_kwargs
            item["fit_info"]["optimizer"] = model.optimizer.get_config()['name']

    _write_json(actual_tt_json, actual_tt_json_path)
