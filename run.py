# You shouldn't need to edit this file, but feel free to take a look at how things are called and run remotely
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import modal

from .image import image, method_name, modal_volumes, sfm_bench_volume

app = modal.App(
    f"sfm-bench-{method_name}",
    image=(
        image  # If using Dockerfile, replace with `modal.Image.from_dockerfile("Dockerfile")`
        # Overwrite build repo (which is only pulled in once for install) with the current local working directory
        .add_local_dir(Path.cwd(), f"/root/{method_name}")
    ),
    volumes=modal_volumes,
)


@contextmanager
def log_max_gpu_memory(log_file: str):
    """Context manager to track GPU memory usage and log maximum memory to a file."""
    import gpu_tracker as gput

    with gput.Tracker(sleep_time=0.1, gpu_ram_unit="megabytes", disable_logs=True) as t:
        yield

    with open(log_file, "w") as f:
        f.write(str(int(t.resource_usage.max_gpu_ram.system)))  # type: ignore


@contextmanager
def log_time(log_file: str):
    """Context manager to track execution time and log it to a file."""
    start_time = time.time()
    yield
    duration = time.time() - start_time

    with open(log_file, "w") as f:
        f.write(f"{duration:.2f}\n")


@app.function(
    timeout=3600 * 8,
    gpu="A100-80GB",
)
def eval(data: str):
    data_folder = Path(f"/sfm-bench/data/{data}/")
    output_folder = Path(f"/sfm-bench/methods/{method_name}/{data}/")

    # Download from gcs (noop if already exists)
    os.system(f"mkdir -p {data_folder}/")
    os.system(f"gsutil -m rsync -r -d gs://nvs-bench/data/{data} {data_folder}")

    # Clean output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    images_folder = output_folder / "images"
    sparse_gt_folder = output_folder / "sparse_gt"
    sparse_folder = output_folder / "sparse"
    os.system(f"rsync -r -d {data_folder}/images/ {images_folder}")
    os.system(f"rsync -r -d {data_folder}/sparse/ {sparse_gt_folder}")
    shutil.rmtree(sparse_folder, ignore_errors=True)

    with log_max_gpu_memory(f"{output_folder}/max_gpu_memory.txt"), log_time(f"{output_folder}/time.txt"):
        subprocess.run(f"bash sfm-bench/eval.sh {output_folder}", shell=True, check=True)

    sfm_bench_volume.commit()


def full_eval():
    """Runs without waiting for each scene to finish"""
    BENCHMARK_DATA = [  # noqa: N806
        # Mipnerf360
        "mipnerf360/bicycle",
        "mipnerf360/treehill",
        "mipnerf360/stump",
        "mipnerf360/room",
        "mipnerf360/kitchen",
        "mipnerf360/garden",
        "mipnerf360/flowers",
        "mipnerf360/counter",
        "mipnerf360/bonsai",
        # Tanks and Temples
        "tanksandtemples/truck",
        "tanksandtemples/train",
        # DeepBlending
        "deepblending/playroom",
        "deepblending/drjohnson",
        # ZipNerf
        "zipnerf/alameda",
        "zipnerf/berlin",
        "zipnerf/london",
        "zipnerf/nyc",
    ]

    eval.for_each(BENCHMARK_DATA, ignore_exceptions=True)


@app.local_entrypoint()
def main(data: str | None = None):
    """Run train/render on a scene (eg: mipnerf360/bicycle) or if not provided the full eval"""
    if data is not None:
        # Assert there's only one / in the data
        assert data.count("/") == 1, "data must be in the format <dataset>/<scene>"
        eval.remote(data)
    else:
        full_eval()
