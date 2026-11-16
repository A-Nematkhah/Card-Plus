"""GST checkpoint paths and download for CrowdNav++ / EvoNav parity."""
from __future__ import annotations

import os
import urllib.request
from typing import Iterable, List, Sequence, Tuple

from code_generation.self_reflection.crowdnav_env_utils import CROWDNAV_ROOT

GST_RAND_NAME = (
    "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-"
    "edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand"
)
GST_NORAND_NAME = (
    "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-"
    "edge_head_0-ebd_64-snl_1-snh_8-seed_1000"
)

_GITHUB_RAW = (
    "https://raw.githubusercontent.com/Shuijing725/CrowdNav_Prediction_AttnGraph/main/"
    "gst_updated/results"
)

# Minimum files VecPretextNormalize / CrowdNavPredInterface need.
_REQUIRED_REL: Tuple[str, ...] = (
    "checkpoint/args.pickle",
    "checkpoint/epoch_100.pt",
)


def gst_model_dirname(*, randomized: bool = True) -> str:
    return GST_RAND_NAME if randomized else GST_NORAND_NAME


def gst_model_dir(*, randomized: bool = True) -> str:
    """Absolute path to the ``.../sj`` directory expected by ``pred.model_dir``."""
    name = gst_model_dirname(randomized=randomized)
    return os.path.join(CROWDNAV_ROOT, "gst_updated", "results", name, "sj")


def gst_checkpoint_ready(*, randomized: bool = True) -> bool:
    root = gst_model_dir(randomized=randomized)
    return all(os.path.isfile(os.path.join(root, rel)) for rel in _REQUIRED_REL)


def _download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def download_gst_checkpoint(
    *,
    randomized: bool = True,
    force: bool = False,
    extra_files: Sequence[str] = (),
) -> str:
    """Download upstream GST weights into the vendored CrowdNav tree.

    Returns the absolute ``sj`` model directory.
    """
    name = gst_model_dirname(randomized=randomized)
    dest_root = gst_model_dir(randomized=randomized)
    rels: List[str] = list(_REQUIRED_REL) + list(extra_files)
    for rel in rels:
        dest = os.path.join(dest_root, rel)
        if os.path.isfile(dest) and not force:
            continue
        url = f"{_GITHUB_RAW}/{name}/sj/{rel}"
        _download_file(url, dest)
    if not gst_checkpoint_ready(randomized=randomized):
        raise FileNotFoundError(
            f"GST checkpoint incomplete under {dest_root}. "
            f"Need files: {', '.join(_REQUIRED_REL)}"
        )
    return dest_root


def ensure_gst_checkpoint(*, randomized: bool = True, download: bool = True) -> str:
    if gst_checkpoint_ready(randomized=randomized):
        return gst_model_dir(randomized=randomized)
    if not download:
        raise FileNotFoundError(
            f"GST checkpoint missing at {gst_model_dir(randomized=randomized)}. "
            "Run download_gst_checkpoint() or pass download=True."
        )
    return download_gst_checkpoint(randomized=randomized)


if __name__ == "__main__":
    path = download_gst_checkpoint(randomized=True)
    print(f"GST ready at: {path}")
