import platform
import subprocess

import tensorflow as tf

from TTBenchmark.constant import SUPPORT_GPU_TYPES


def check_env_info():
    gpu_found = True
    try:
        sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        gpu_found = False

    env = {}
    if gpu_found:
        out_str = sp.communicate()
        out_list = str(out_str[0]).split('\\n')

        split_res = out_list[2].split()
        assert split_res[3].lower() == "driver"
        env["drive_v"] = split_res[5]
        assert split_res[6].lower() == "cuda"
        env["cuda_v"] = split_res[8]
        split_res = out_list[8].split()
        gpu_type = "".join([x for x in "_".join(split_res[2:4]).lower() if x != '.'])
        assert gpu_type in SUPPORT_GPU_TYPES
        env["gpu_type"] = gpu_type
    env["tf_v"] = tf.__version__
    env["cpu"] = platform.processor()

    return env


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True
