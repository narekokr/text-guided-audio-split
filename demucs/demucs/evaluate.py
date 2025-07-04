# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging

from dora.log import LogProgress
import numpy as np
# import musdb
import museval
import torch as th

from .apply import apply_model
from .audio import convert_audio, save_audio
from . import distrib
from .utils import DummyPoolExecutor


logger = logging.getLogger(__name__)


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores


def eval_track(references, estimates, win, hop, compute_sdr=True):
    references = references.transpose(1, 2).double()
    estimates = estimates.transpose(1, 2).double()

    new_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]

    if not compute_sdr:
        return None, new_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores


def evaluate(solver, compute_sdr=False):
    args = solver.args

    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # -- CHANGE: use your PT files instead of MUSDB --
    pt_dir = Path(args.dset.valid_pt_dir)  # or args.dset.valid_dir etc
    pt_files = sorted(pt_dir.glob("*.pt"))

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)
    eval_device = solver.device if hasattr(solver, "device") else "cpu"

    tracks = {}
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location=eval_device)
        mix = data["mix"]                # shape: [C, L] or [1, C, L]
        references = data["stem"]        # shape: [S, C, L] or [C, L]
        conditioning = data["clap_embedding"]  # shape: [S, D] or [D]

        # --- Shape sanity checks ---
        if mix.dim() == 2:
            mix = mix.unsqueeze(0)  # [1, C, L]
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)  # [1, D] (if single stem)
        # Move to device
        mix = mix.to(eval_device)
        references = references.to(eval_device)
        conditioning = conditioning.to(eval_device)

        # --- Run model ---
        with torch.no_grad():
            estimates = apply_model(model, mix, conditioning=conditioning)[0]  # [S, C, L] (check!)

        # --- Compute metrics ---
        # You may need to match shape [S, C, L] for references and estimates
        if references.dim() == 2:
            references = references.unsqueeze(0)
        if estimates.dim() == 2:
            estimates = estimates.unsqueeze(0)
        win_len = win
        hop_len = hop
        # SDR calculation
        _, nsdrs = eval_track(references, estimates, win=win_len, hop=hop_len, compute_sdr=compute_sdr)
        # Store result
        tracks[pt_file.stem] = {f"nsdr_{i}": float(n) for i, n in enumerate(nsdrs)}

    # -- Aggregate results (mean NSDR over all stems/tracks) --
    all_nsdrs = [val for t in tracks.values() for val in t.values()]
    result = {"nsdr_mean": float(np.mean(all_nsdrs))}
    return result
