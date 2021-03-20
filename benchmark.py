from typing import List
from typing import NamedTuple
from zipfile import ZipFile

import requests
import tensorflow as tf
from tensorflow.keras.applications import VGG16

from benchmark_constants import GPU_STATS

# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications import ResNet50V2

# from tensorflow.keras.applications import Xception


class BenchmarkData(NamedTuple):
    data_name: str
    actual_time: float
    raw_model: tf.keras.Model
    trained_model_url: str
    batch_size: int
    optimizer: str
    training_size: int
    validation_split: float


class BenchmarkConfig(NamedTuple):
    gpu_type: str
    tf_version: str
    benchmark_data: List[BenchmarkData]


BENCHMARK = [
    BenchmarkConfig(
        gpu_type="T4",
        tf_version="2.4.1",
        benchmark_data=[
            BenchmarkData(
                data_name="vgg16_input32_batch4_optimizer-sgd",
                actual_time=21.59,
                raw_model=VGG16,
                trained_model_url="https://benchmark-models.s3.amazonaws.com/vgg16_input32_batch4_optimizer-sgd.zip",
                batch_size=4,
                optimizer="sgd",
                training_size=5000,
                validation_split=0.1,
            ),
            BenchmarkData(
                data_name="vgg16_input32_batch4_optimizer-adam",
                actual_time=24.97,
                raw_model=VGG16,
                trained_model_url="https://benchmark-models.s3.amazonaws.com/vgg16_input32_batch4_optimizer-adam.zip",
                batch_size=4,
                optimizer="adam",
                training_size=5000,
                validation_split=0.1,
            ),
        ],
    )
]


def run_benchmark(
    gpu_type: str,
    tf_version: str,
    model_file: str,
    scaler: str,
    parse_cnn_func,
    predictor_func,
):
    results = {}
    for bmconfig in BENCHMARK:
        if bmconfig.gpu_type == gpu_type and bmconfig.tf_version == tf_version:
            for bmdata in bmconfig.benchmark_data:
                trained_model = _download_saved_model_and_load(
                    bmdata.trained_model_url, bmdata.data_name
                )
                parsed_model = parse_cnn_func(trained_model.layers)
                layer_names, layer_predictions = predictor_func(
                    parsed_model,
                    bmdata.batch_size,
                    bmdata.optimizer,
                    GPU_STATS[bmconfig.gpu_type],
                    model_file,
                    scaler,
                )
                iterations = (bmdata.training_size / bmdata.batch_size) * (
                    1 - bmdata.validation_split
                )
                predicted_time = round(sum(layer_predictions) * iterations / 1000, 2)
                results[bmdata.data_name] = {
                    "actual_time": bmdata.actual_time,
                    "predicted_time": predicted_time,
                }
    return results


def _download_saved_model_and_load(url: str, model_name):
    r = requests.get(url)
    zip_name = f"{model_name}.zip"
    open(zip_name, "wb").write(r.content)
    with ZipFile(zip_name, "r") as zipObj:
        zipObj.extractall()
        trained_model = tf.keras.models.load_model(model_name)
        return trained_model
