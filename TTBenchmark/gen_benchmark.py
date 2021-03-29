import collections
import json
import os

import numpy as np
import tensorflow as tf
from model_data_util.create_tt_data.model_data_convert import convertModelToRawData, convertRawDataToModel

from TTBenchmark.check_environment import check_env_info, in_notebook
from TTBenchmark.constant import GDRIVE_PATH

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BenchmarkDataMini():
    def __init__(self,
                 raw_model: tf.keras.Model,
                 actual_tt_mean: float,
                 actual_tt_median: float,
                 actual_tt_std: float,
                 batch_size: int,
                 validation_split: float,
                 x_shape: np.array,
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

        self.data: dict = {"x_shape": x_shape}


def _ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def _write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def _get_benchmark_path(gdrive_path, model_type, fname="trained_tt.json"):
    env_fname = "_".join(list(check_env_info().values()))
    env_path = os.path.join(gdrive_path, env_fname)
    actual_tt_json_path = os.path.join(env_path, model_type, fname)
    return actual_tt_json_path


def save_benchmark(
        benchmarks_mini: list,
        columns: list,
        model_type: str,
        gdrive_path=GDRIVE_PATH,
        replace=True
):
    """
    save benchmarks into json structure:
        model_name1:{
            model_df: df,
            actual_tt: {mean: , median: , std:}
            fit_kwargs: {batch_size: , validation_split: , verbose: }
        }
    :param benchmarks_mini: list of benchmark_mini
    :param columns: 
    :param model_type: 
    :param gdrive_path: 
    :return: 
    """
    actual_tt_json_path = _get_benchmark_path(gdrive_path, model_type)
    _ensure_dir(actual_tt_json_path)
    model_index = None
    if not os.path.exists(actual_tt_json_path) or replace == True:
        actual_tt_json = collections.defaultdict(dict)
        _write_json(actual_tt_json, actual_tt_json_path)
        model_index = 0

    with open(actual_tt_json_path) as f:
        actual_tt_json = json.load(f)
        for i, bmdatamini in enumerate(tqdm(benchmarks_mini)):
            model = bmdatamini.model_info["raw_model"]
            actual_tt = bmdatamini.actual_tt
            fit_kwargs = bmdatamini.fit_kwargs
            x_shape = bmdatamini.data['x_shape']

            if model_index is None and i == 1:
                all_models = list(actual_tt_json.keys())
                if len(all_models) > 0:
                    model_index = int(max(all_models).split("_")[-1]) + 1
                else:
                    # empty json file
                    model_index = 0
            else:
                model_index += 1

            model_name = f"{model_type}_{model_index}"

            actual_tt_json[model_name] = {}

            batch_input_shape = np.array([fit_kwargs["batch_size"], *x_shape[1:]])
            num_data = x_shape[0]

            item = actual_tt_json[model_name]
            item["model_df"] = convertModelToRawData(model, columns, num_data, batch_input_shape,
                                                     num_dim=len(
                                                         batch_input_shape)).to_dict()  # with no out_dim padding
            item["actual_tt"] = actual_tt
            item["fit_kwargs"] = fit_kwargs

    _write_json(actual_tt_json, actual_tt_json_path)


if __name__ == "__main__":
    import pickle
    import pandas as pd

    res = pickle.load(open(
        "/Users/wangqiong/Documents/AIpaca/Code/TT Prediction/benchmark/benchmark_lib/local_data/ffnn_data_V100_benchmark.pkl",
        "rb"))

    benchmarks_mini = []
    res['X_df'] = [pd.DataFrame(x_df) for x_df in res['X_df']]
    columns = res['X_df'][0].columns
    model_type = "ffnn_dense_only"
    for i, X_df in enumerate(tqdm(res['X_df'][:10])):
        X_df = pd.DataFrame(X_df)
        model, training_size, batch_input_shape = convertRawDataToModel(X_df)
        x_shape = np.array([training_size, *batch_input_shape[1:]])
        assert len(X_df.out_dim_0.unique()) == 1
        bmdata_mini = BenchmarkDataMini(
            raw_model=model,
            actual_tt_mean=res['y_mean'][i],
            actual_tt_median=res['y_median'][i],
            actual_tt_std=res['y_std'][i],
            batch_size=int(batch_input_shape[0]),
            validation_split=0,
            x_shape=x_shape,
            verbose=False
        )
        benchmarks_mini.append(bmdata_mini)
    save_benchmark(benchmarks_mini, columns, model_type,
                   gdrive_path="/Users/wangqiong/Documents/AIpaca/Code/TT Prediction/benchmark/benchmark_lib/local_data",
                   replace=True)
