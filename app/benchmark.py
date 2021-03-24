import json
from typing import List
from typing import NamedTuple

import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception

MODELS = {
    "vgg16": VGG16,
    "mobilenet_1": MobileNet,
    "resnet50v2": ResNet50V2,
    "xception": Xception,
    "inception_resnet_v2": InceptionResNetV2,
    "inception_v3": InceptionV3,
}
GPU_STATS = {
    "T4": {"bandwidth": 298.08, "cores": 40, "clock": 1590},
    "V100": {"bandwidth": 900, "cores": 5120, "clock": 1455},
    "P100": {"bandwidth": 732, "cores": 3584, "clock": 1303},
    "K80": {"bandwidth": 240, "cores": 2496, "clock": 875},
}
GDRIVE_PATH = "/content/drive/MyDrive/benchmark"


class BenchmarkData(NamedTuple):
    data_name: str
    actual_time: float
    raw_model: tf.keras.Model
    trained_model_path: str
    batch_size: int
    optimizer: str
    training_size: int
    validation_split: float


def run_benchmark(
    gpu_type: str,
    tf_version: str,
    model_file: str,
    scaler: str,
    parse_cnn_func,
    predictor_func,
):
    actual_tt_json_path = f"{GDRIVE_PATH}/{gpu_type}-{tf_version}/trained_tt.json"
    benchmark = _load_benchmark(gpu_type, tf_version, actual_tt_json_path)

    results = {}
    for bmdata in benchmark:
        trained_model = trained_model = tf.keras.models.load_model(
            bmdata.trained_model_path
        )
        parsed_model = parse_cnn_func(trained_model.layers)
        layer_names, layer_predictions = predictor_func(
            parsed_model,
            bmdata.batch_size,
            bmdata.optimizer,
            GPU_STATS[gpu_type],
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


def _load_benchmark(
    gpu_type: str, tf_version: str, actual_tt_json_path: str
) -> List[BenchmarkData]:

    with open(actual_tt_json_path) as f:
        actual_tt_json = json.load(f)

    benchmark = []
    for model_name, raw_model in MODELS.items():
        if model_name not in actual_tt_json:
            continue
        train_tt_model_data = actual_tt_json[model_name]
        for data_name, stats in train_tt_model_data.items():
            bmdata = BenchmarkData(
                data_name=data_name,
                actual_time=stats["actual_time"],
                raw_model=raw_model,
                trained_model_path=f"{GDRIVE_PATH}/{gpu_type}-{tf_version}/saved_models/{data_name}",
                batch_size=stats["batch_size"],
                optimizer=stats["optimizer"],
                training_size=stats["training_size"],
                validation_split=stats["validation_split"],
            )
            benchmark.append(bmdata)

    return benchmark
