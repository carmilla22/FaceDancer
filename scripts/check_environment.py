#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("FACEDANCER_ROOT", Path(__file__).resolve().parents[1])
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MINIMUM_DRIVER = (525, 60, 13)
EXPECTED_GPU_NAME = "RTX 4500 Ada"
EXPECTED_COMPUTE_CAPABILITY = (8, 9)


def fail(message):
    raise RuntimeError(message)


def version_tuple(version):
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as error:
        raise RuntimeError(
            "Could not parse NVIDIA driver version: {}".format(version)
        ) from error


def check_driver():
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        fail("nvidia-smi is not available on PATH")

    libcuda = Path("/run/opengl-driver/lib/libcuda.so.1")
    if not libcuda.is_file():
        fail("NixOS host driver library is missing: {}".format(libcuda))

    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first_gpu = result.stdout.strip().splitlines()[0]
    try:
        gpu_name, driver_version = (
            field.strip() for field in first_gpu.rsplit(",", 1)
        )
    except ValueError as error:
        raise RuntimeError(
            "Unexpected nvidia-smi output: {}".format(first_gpu)
        ) from error

    if version_tuple(driver_version) < MINIMUM_DRIVER:
        fail(
            "NVIDIA driver {} is older than CUDA 12.2's minimum {}".format(
                driver_version, ".".join(map(str, MINIMUM_DRIVER))
            )
        )
    print("[ok] NVIDIA driver {}: {}".format(driver_version, gpu_name))


def check_tensorflow():
    if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        fail("TF_USE_LEGACY_KERAS must be set to 1 before importing TensorFlow")

    import tensorflow as tf
    import tf_keras

    if tf.keras.Model is not tf_keras.Model:
        fail("TensorFlow is not using the installed legacy tf-keras package")

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        fail("TensorFlow did not discover a physical GPU")

    details = tf.config.experimental.get_device_details(gpus[0])
    device_name = details.get("device_name")
    if device_name and EXPECTED_GPU_NAME not in device_name:
        fail(
            "Expected {}, but TensorFlow reported {}".format(
                EXPECTED_GPU_NAME, device_name
            )
        )

    compute_capability = details.get("compute_capability")
    if (
        compute_capability
        and tuple(compute_capability) != EXPECTED_COMPUTE_CAPABILITY
    ):
        fail(
            "Expected compute capability 8.9, but TensorFlow reported {}.{}".format(
                *compute_capability
            )
        )

    with tf.device("/GPU:0"):
        left = tf.random.uniform((1024, 1024), dtype=tf.float32)
        right = tf.random.uniform((1024, 1024), dtype=tf.float32)
        product = tf.matmul(left, right)
    product.numpy()
    if "GPU:0" not in product.device:
        fail("Explicit matrix multiplication did not execute on /GPU:0")

    reported_name = device_name or gpus[0].name
    reported_capability = (
        ".".join(map(str, compute_capability))
        if compute_capability
        else "not exposed"
    )
    print(
        "[ok] TensorFlow {} legacy tf-keras GPU matmul: {} (compute {})".format(
            tf.__version__, reported_name, reported_capability
        )
    )


def check_mediapipe():
    import mediapipe as mp

    from utils.hand_occlusion import create_hand_landmarker

    model_path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    landmarker = create_hand_landmarker(model_path)
    landmarker.close()
    print(
        "[ok] MediaPipe {} Hand Landmarker: {}".format(
            mp.__version__, model_path
        )
    )


def main():
    try:
        check_driver()
        check_tensorflow()
        check_mediapipe()
    except Exception as error:
        print("[failed] {}".format(error), file=sys.stderr)
        return 1

    print("FaceDancer GPU environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
