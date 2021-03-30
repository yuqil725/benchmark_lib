import collections
import json
import os
import pickle
from time import time
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from model_data_util.create_tt_data.model_data_convert import convertRawDataToModel, convertModelToRawData, \
    preprocessRawData

from TTBenchmark.check_environment import check_env_info, in_notebook
from TTBenchmark.constant import GDRIVE_PATH

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BenchmarkData:
    # model
    def __init__(self):
        self.model_info: dict = {}
        self.model_info["model_name"]: str
        self.model_info["raw_model"]: tf.keras.Model
        # tt
        self.actual_tt: dict = {}
        self.actual_tt["mean"]: float
        self.actual_tt["median"]: float
        self.actual_tt["std"]: float
        # fit_info
        self.fit_kwargs: dict = {}
        self.fit_kwargs["batch_size"]: int
        self.fit_kwargs["validation_split"]: float
        self.fit_kwargs["verbose"]: bool
        # data
        self.data: dict = {}
        self.data["x_shape"]: np.array


def run_benchmark(
        tt_predictor: tf.keras.Model,
        get_feature_func,
        kwargs: dict = {},
        model_types: list = None,
        predict_type: str = "median",
        gdrive_path=GDRIVE_PATH,
        env_path=None,
        upper_limit_per_type=None
) -> dict:
    """
    :param tt_predictor: a model can predict training time
    :param get_feature_func: a function used to grab features from tensorflow models
    :param kwargs: by default, all arguments (except get_feature_func and kwargs) passed into run_benchmarks will be added into kwargs;
    additional argument such as batch_input_shape will also be added
    :param model_types: a list of model categories. If None, all models in the matched environment benchmark will be tested
    :param predict_type: must be one of the value "median", "mean", "std"
    :param gdrive_path: the path to the benchmark model file, in default is "/content/drive/MyDrive/benchmark"
    :param env_path: environment directory path. Only specify value if not testing the current environment
    :param upper_limit_per_type: the maximum number of benchmark acceptable per model type
    :return: benchmark data object
    """
    if env_path is None:
        env_fname = "_".join(list(check_env_info().values()))
        env_path = os.path.join(gdrive_path, env_fname)
    if model_types is None:
        model_types = [x for x in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, x))]

    result = collections.defaultdict(dict)
    kwargs["tt_predictor"] = tt_predictor
    kwargs["predict_type"] = predict_type
    kwargs["gdrive_path"] = gdrive_path
    for model_type in model_types:
        print(f"Testing model type: {model_type}")
        actual_tt_json_path = os.path.join(env_path, model_type, "trained_tt.json")
        benchmarks = _load_benchmark(actual_tt_json_path, upper_limit_per_type)
        kwargs["model_type"] = model_type
        kwargs["actual_tt_json_path"] = actual_tt_json_path
        print("Start test...")
        limit = upper_limit_per_type if upper_limit_per_type is not None else len(benchmarks)
        for bmmodel in tqdm(benchmarks[:limit]):
            kwargs["batch_size"] = bmmodel.fit_kwargs["batch_size"]
            X = get_feature_func(bmmodel.model_info["raw_model"], bmmodel.data["x_shape"], kwargs)
            y_pred = tt_predictor.predict(X).flatten()
            result[model_type][bmmodel.model_info["model_name"]] = {}
            result[model_type][bmmodel.model_info["model_name"]]["tt_pred"] = y_pred
            assert predict_type in bmmodel.actual_tt, f"predict_type must be one of the value {list(bmmodel.actual_tt.keys())}"
            result[model_type][bmmodel.model_info["model_name"]]["tt_actual"] = bmmodel.actual_tt[predict_type]

    return result


def _load_benchmark(
        actual_tt_json_path: str,
        upper_limit_per_type: int
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
    print("loading benchmarks...")
    if upper_limit_per_type is None:
        upper_limit_per_type = len(list(actual_tt_json.keys()))
    for model_name, stat in tqdm(list(actual_tt_json.items())[:upper_limit_per_type]):
        bmdata = BenchmarkData()
        bmdata.fit_kwargs = stat["fit_kwargs"]
        bmdata.actual_tt = stat["actual_tt"]
        bmdata.model_info["raw_model"], training_size, batch_input_shape = convertRawDataToModel(
            pd.DataFrame(stat["model_df"]))
        bmdata.data["x_shape"] = np.array([training_size, *batch_input_shape[1:]])
        bmdata.model_info["model_name"] = model_name
        benchmarks.append(bmdata)
    return benchmarks


if __name__ == "__main__":
    def get_feature_func(model, x_shape, kwargs):
        time0 = time()
        training_size = x_shape[0]
        actual_tt_json_path = kwargs["actual_tt_json_path"]
        actual_tt_json = json.load(open(actual_tt_json_path))
        random_key = list(actual_tt_json.keys())[0]
        columns = list(actual_tt_json[random_key]["model_df"].keys())

        padding = kwargs["tt_predictor"].layers[0].input_shape[1]

        batch_input_shape = np.array([kwargs["batch_size"], *x_shape[1:]])
        df = convertModelToRawData(model, columns, training_size, batch_input_shape)
        X = preprocessRawData(df, kwargs["one_hot_enc"], padding).values
        X = X.reshape((-1, *X.shape))
        return X


    model_type = "ffnn_dense_only"
    gdrive_path = GDRIVE_PATH
    model_path = "/Users/wangqiong/Documents/AIpaca/Code/TT Prediction/benchmark/benchmark_lib/local_data/model"
    gdrive_path = "/Users/wangqiong/Documents/AIpaca/Code/TT Prediction/benchmark/benchmark_lib/local_data"
    tt_predictor = tf.keras.models.load_model(model_path)
    one_hot_enc = pickle.load(open(os.path.join(model_path, "one_hot_enc.pkl"), "rb"))
    kwargs = {"one_hot_enc": one_hot_enc}
    benchmarks = run_benchmark(tt_predictor, get_feature_func, gdrive_path=gdrive_path, kwargs=kwargs,
                               upper_limit_per_type=None)
    print(benchmarks)
