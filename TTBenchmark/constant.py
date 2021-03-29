# MODELS = {
#     "vgg16": VGG16,
#     "mobilenet_1": MobileNet,
#     "resnet50v2": ResNet50V2,
#     "xception": Xception,
#     "inception_resnet_v2": InceptionResNetV2,
#     "inception_v3": InceptionV3,
# }
# GPU_STATS = {
#     "T4": {"bandwidth": 298.08, "cores": 40, "clock": 1590},
#     "V100": {"bandwidth": 900, "cores": 5120, "clock": 1455},
#     "P100": {"bandwidth": 732, "cores": 3584, "clock": 1303},
#     "K80": {"bandwidth": 240, "cores": 2496, "clock": 875},
# }
GDRIVE_PATH = "/content/drive/MyDrive/benchmark"

SUPPORT_GPU_TYPES = {"tesla_k40", "tesla_k80", "tesla_p100-pcie", "tesla_t4", "tesla_v100", "tesla_v100-sxm2"}
