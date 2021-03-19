from typing import List
from typing import NamedTuple

import tensorflow as tf
from tensorflow.keras.applications import VGG16

# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications import ResNet50V2

# from tensorflow.keras.applications import Xception

GPU_STATS = {"T4": {"bandwidth": 298.08, "cores": 40, "clock": 1590}}


class BenchmarkData(NamedTuple):
    data_name: str
    actual_time: float
    raw_model: tf.keras.Model
    trained_model_path: str
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
                data_name="vgg16_input128_batch4_optimizer-sgd",
                actual_time=41.77,
                raw_model=VGG16,
                trained_model_path="./saved_models/vgg16_input128_batch4_optimizer-sgd",
                batch_size=4,
                optimizer="sgd",
                training_size=5000,
                validation_split=0.1,
            ),
            # BenchmarkData(
            #     data_name="vgg16_input128_batch4_optimizer-adam",
            #     actual_time=41.77,
            #     raw_model=VGG16,
            #     trained_model_path='',
            #     batch_size=4,
            #     optimizer='adam',
            #     training_size=5000,
            #     validation_split=0.1
            # ),
            # BenchmarkData(
            #     data_name="mobilenet_1_input128_batch4_optimizer-sgd",
            #     actual_time=15.92,
            #     raw_model=MobileNet,
            #     trained_model_path='',
            #     batch_size=4,
            #     optimizer='sgd',
            #     training_size=5000,
            #     validation_split=0.1
            # )
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
                trained_model = tf.keras.models.load_model(bmdata.trained_model_path)
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
                predicted_time = sum(layer_predictions) * iterations / 1000
                results[bmdata.data_name] = {
                    "actual_time": bmdata.actual_time,
                    "predicted_time": predicted_time,
                }
    return results


# train_t4_tf241_hist = {
#     "gpu_type": "T4",
#     "tf_version": "2.4.1",
#     "vgg16": {
#         "vgg16_input128_batch4_optimizer-sgd": 41.77,
#         "vgg16_input128_batch4_optimizer-adam": 48.74,
#     },
#     "mobilenet_1": {
#         "mobilenet_1_input128_batch4_optimizer-sgd": 15.43,
#         "mobilenet_1_input128_batch4_optimizer-adam": 15.92,
#     },
#     "xception": {
#         "xception_input128_batch4_optimizer-sgd": 43.28,
#         "xception_input128_batch4_optimizer-adam": 46.04,
#     },
#     "resnet50v2": {
#         "resnet50v2_input128_batch4_optimizer-sgd": 38.76,
#         "resnet50v2_input128_batch4_optimizer-adam": 41.76,
#     },
#     "inception_v3": {
#         "inception_v3_input128_batch4_optimizer-sgd": 37.92,
#         "inception_v3_input128_batch4_optimizer-adam": 40.53,
#     },
# }
