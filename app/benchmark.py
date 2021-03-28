import json
import os
from typing import List
from typing import NamedTuple

import tensorflow as tf
from model_data_util.create_tt_data.model_data_convert import convertRawDataToModel

from check_environment import check_env_info
from constant import GDRIVE_PATH


class BenchmarkData(NamedTuple):
    # model
    model_info: dict
    model_info["model_name"]: str
    model_info["raw_model"]: tf.keras.Model
    # tt
    actual_tt: dict
    actual_tt["mean"]: float
    actual_tt["median"]: float
    actual_tt["std"]: float
    # fit_info
    fit_kwargs: dict
    fit_kwargs["batch_size"]: int
    fit_kwargs["optimizer"]: str
    fit_kwargs["validation_split"]: float
    fit_kwargs["verbose"]: bool
    # data
    training_size: int


def run_benchmark(
        tt_predictor: tf.keras.Model,
        get_feature_func,
        model_types: list = None,
        predict_type: str = "median",
        gdrive_path=GDRIVE_PATH,
) -> dict:
    """
    :param tt_predictor: a model can predict training time
    :param get_feature_func: a function used to grab features from tensorflow models
    :param model_types: a list of model categories. If None, all models in the matched environment benchmark will be tested
    :param predict_type: must be one of the value "median", "mean", "std"
    :param gdrive_path: the path to the benchmark model file, in default is "/content/drive/MyDrive/benchmark"
    :return: benchmark data object
    """
    env_fname = "_".join(list(check_env_info().values()))
    env_path = os.path.join(gdrive_path, env_fname)
    if model_types is None:
        model_types = [x for x in os.listdir(env_path) if os.path.isdir(x)]
    for model_type in model_types:
        actual_tt_json_path = os.path.join(env_path, model_type, "trained_tt.json")
        benchmarks = _load_benchmark(actual_tt_json_path)

    result = {}
    for bmmodel in benchmarks:
        X = get_feature_func(bmmodel.model_info["raw_model"], bmmodel.training_size)
        y_pred = tt_predictor.predict(X)
        result[bmmodel.model_info["model_name"]] = {}
        result[bmmodel.model_info["model_name"]]["tt_pred"] = y_pred
        assert predict_type in bmmodel.actual_tt, f"predict_type must be one of the value {list(bmmodel.actual_tt.keys())}"
        result[bmmodel.model_info["model_name"]]["tt_actual"] = bmmodel.actual_tt[predict_type]

    return result


def _load_benchmark(
        actual_tt_json_path: str
) -> List[BenchmarkData]:
    """
    json structure:
        model_name1:{
            model_df: df,
            actual_tt: {mean: , median: , std:}
            fit_kwargs: {batch_size: , optimizer: , validation_split: , verbose: }
        }

    :param actual_tt_json_path:
    :return:
    """
    with open(actual_tt_json_path) as f:
        actual_tt_json = json.load(f)

    benchmarks = []
    for model_name, stat in actual_tt_json.items():
        bmdata = BenchmarkData()
        assert list(bmdata.fit_kwargs.keys()) == list(stat["fit_kwargs"].keys())
        bmdata.fit_kwargs = stat["fit_kwargs"]
        assert list(bmdata.actual_tt.keys()) == list(stat["actual_tt"].keys())
        bmdata.actual_tt = stat["actual_tt"]
        bmdata.model_info["raw_model"], bmdata.training_size = convertRawDataToModel(stat["model_df"])
        bmdata.model_info["model_name"] = model_name
        benchmarks.append(bmdata)
    return benchmarks
