# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor-parallel numerical-parity harness for the MammothModa2 AR stage.

Refs #7114 (AR stage -> Parallelism -> "Validate tensor parallelism (TP),
including generation-specific parameters and stage-output correctness"),
tracked under #7075.

Why this needs its own harness
------------------------------
The AR stage carries three MammothModa2-specific pieces that upstream vLLM's
TP paths never exercise:

* An **extra generation vocabulary** (``gen_vocab_size=32800``) with its own
  ``VocabParallelEmbedding`` / ``ParallelLMHead`` (``gen_embed_tokens`` /
  ``gen_head``).  ``MammothModa2Qwen2ForCausalLM.compute_logits`` concatenates
  the two logit tensors: ``torch.cat([base_logits, gen_logits], dim=-1)``.
  32800 is not a multiple of ``DEFAULT_VOCAB_PADDING_SIZE``, so the padded and
  sharded width differs from the logical width -- if the padding is not sliced
  off before the concat, the gen half lands at the wrong offset.
* **Per-request t2i token constraints** (``_apply_t2i_token_constraints``) that
  index that concatenated tensor by *absolute* token id (``eol_token_id``,
  ``visual_token_start_id``, ...).  A shifted offset silently constrains the
  wrong columns.
* A **stage output that is a hidden-state tensor**, not text, handed to the DiT
  stage through ``stage_input_processors.mammoth_moda2.ar2dit``.  The
  production code there asserts only the *length* of that tensor.

Every one of those failure modes is silent: an image is still produced, it is
just wrong.  So the checks are ordered strongest-first rather than
end-to-end-first.

Check layers
------------
=====  ==========================  ============================================
L0     ``test_gen_vocab_padding``  Pure config; no GPU.  Pins down how 32800
                                   pads and shards, and asserts the base/gen
                                   split stays representable at every TP.
L0.5   ``test_ar_grid_structure``  Needs no reference run: the emitted grid
                                   must be ``ar_height`` rows of ``ar_width``
                                   visual tokens each closed by ``eol``.  A
                                   shifted concat offset breaks this on its
                                   own, at a single TP degree.
L1     ``test_ar_token_parity``    Generated token ids vs the TP=1 reference.
L2a    ``test_ar_stage_output_``   AR->DiT hidden states: length, token
       ``alignment``               alignment, dtype, no dead tensor.
L2b    ``test_ar_stage_output_``   Hidden states vs the TP=1 reference.
       ``parity``
L3     ``test_t2i_image_parity``   End-to-end pixel backstop.
=====  ==========================  ============================================

Scope
-----
Stage 0 (AR) only.  Stage 1 (DiT) is pinned to TP=1 on one device for every
run, so any difference observed downstream is attributable to the AR stage.
DiT-side parallelism, AR pipeline/data parallelism and ROCm are out of scope
per #7114.

Running
-------
Full sweep (needs 8 cards)::

    pytest tests/e2e/offline_inference/test_mammoth_moda2_tp_parity.py

One degree::

    pytest tests/e2e/offline_inference/test_mammoth_moda2_tp_parity.py -k tp4

Config-only layer, no accelerator needed::

    pytest tests/e2e/offline_inference/test_mammoth_moda2_tp_parity.py \
        -k test_gen_vocab_padding

Each TP degree starts and tears down exactly one engine; the capture is cached
for the module so all layers share a single run per degree.  Engines are never
held concurrently.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub import snapshot_download
from vllm.sampling_params import SamplingParams

from tests.helpers.clean import cleanup_test_environment
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniRunner, get_model_prefix
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.model_extras.mammothmodal2_preview import build_text_to_image_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "bytedance-research/MammothModa2-Preview"
BASE_DEPLOY_CONFIG = get_deploy_config_path("mammoth_moda2.yaml")

# TP=1 is the reference every other degree is compared against.
REFERENCE_TP = 1
TP_DEGREES = (1, 2, 4, 8)

SEED = 42
# Small grid keeps a full sweep affordable; 256/16 = 16 -> a 16x16 token grid.
IMAGE_HEIGHT = IMAGE_WIDTH = 256
AR_PATCH_SIZE = 16
# Two denoising steps: the DiT is held at TP=1 and is not what is under test,
# it only has to be deterministic.
DIT_INFERENCE_STEPS = 2
DIT_GUIDANCE_SCALE = 1.0
DIT_CFG_RANGE = [0.0, 1.0]

PROMPT_TEXT = "A cat sitting on a laptop keyboard"

# Visual placeholder ids the DiT pipeline uses to split text from AR-image
# conditioning; mirrors tests/e2e/offline_inference/test_mammoth_moda2_expansion.py.
_IMAGE_TOKEN_ID = 151655  # "<|image_pad|>"
_VIDEO_TOKEN_ID = 151656  # "<|video_pad|>"
_VISION_START_TOKEN_ID = 151652  # "<|vision_start|>"
_VISION_END_TOKEN_ID = 151653  # "<|vision_end|>"

# Hidden states cross the stage boundary as float32 but are computed in bf16.
# TP changes the summation order inside every RowParallelLinear all-reduce, so
# bit-exactness is not a meaningful requirement -- bf16 carries ~8 mantissa
# bits (eps ~ 7.8e-3) and the error compounds across layers.  These bounds are
# deliberately loose; the assertion that carries the signal is the cosine
# similarity, and the measured numbers are always printed so PR-3 can report
# them per TP degree.  Override while investigating a specific divergence.
HIDDEN_ATOL = float(os.environ.get("MAMMOTH_TP_HIDDEN_ATOL", "2e-2"))
HIDDEN_RTOL = float(os.environ.get("MAMMOTH_TP_HIDDEN_RTOL", "2e-2"))
HIDDEN_MIN_COSINE = float(os.environ.get("MAMMOTH_TP_HIDDEN_MIN_COSINE", "0.9995"))
# Images are uint8-quantized downstream; 1/255 ~ 3.9e-3.
PIXEL_ATOL = float(os.environ.get("MAMMOTH_TP_PIXEL_ATOL", "4e-3"))

# Fixed sampling coordinates: (channel, row_fraction, col_fraction).
_PIXEL_SAMPLE_COORDS = [
    (0, 0.0, 0.0),
    (0, 0.5, 0.5),
    (0, 1.0, 1.0),
    (0, 0.25, 0.75),
    (1, 0.0, 1.0),
    (1, 0.5, 0.0),
    (1, 0.75, 0.25),
    (1, 1.0, 0.5),
    (2, 0.0, 0.5),
    (2, 0.5, 1.0),
    (2, 0.75, 0.75),
    (2, 1.0, 0.0),
]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.parallel,
]


def _tp_param(tp_size: int) -> Any:
    """One pytest param per TP degree, carrying its own ``cards_{n}`` mark.

    ``hardware_marks`` attaches the SKU, the platform mark and a skipif for
    ``device_count() < num_cards``, so a 2-card box collects the whole sweep
    and skips tp4/tp8 instead of erroring.  A100 is not a registered SKU in
    ``pyproject.toml``; H100 is the closest registered CUDA resource and
    matches the existing MammothModa2 e2e test.
    """
    return pytest.param(
        tp_size,
        marks=hardware_marks(res={"cuda": "H100"}, num_cards=tp_size),
        id=f"tp{tp_size}",
    )


TP_PARAMS = [_tp_param(tp) for tp in TP_DEGREES]
# Comparison layers need the TP=1 reference too, but that run needs 1 card and
# is covered by the ``num_cards=tp_size`` mark of the degree under test.
COMPARISON_TP_PARAMS = [_tp_param(tp) for tp in TP_DEGREES if tp != REFERENCE_TP]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ARCapture:
    """Everything one run of the pipeline tells us about the AR stage."""

    tp_size: int
    #: Token ids the AR stage emitted, excluding the final token (``ar2dit``
    #: drops it: it has no corresponding hidden state).
    generated_token_ids: tuple[int, ...]
    #: prompt + generated ids, as handed to the DiT stage.
    full_token_ids: tuple[int, ...]
    #: Index in ``full_token_ids`` where the generated span starts.
    answer_start_index: int
    #: (num_tokens, hidden_size) float32 on CPU -- the AR -> DiT stage output.
    hidden_states: torch.Tensor
    #: Sampled pixels of the final image, or ``None`` when the DiT produced none.
    image_pixels: tuple[float, ...] | None


@contextmanager
def _record_stage_output(sink: list[dict[str, Any]]) -> Iterator[None]:
    """Record what the AR stage hands to the DiT stage.

    ``stage_init_utils`` resolves ``custom_process_input_func`` eagerly with
    ``getattr(import_module(mod_path), fn_name)`` while the stage is built, so
    the patch has to be installed *before* the runner is constructed.  The
    orchestrator then calls the resolved function in-process, which is why a
    plain module-attribute swap is enough -- no worker-side hook is needed.

    The wrapper delegates to the original, so this observes the production path
    including its own length assertion rather than replacing it.
    """
    import vllm_omni.model_executor.stage_input_processors.mammoth_moda2 as proc

    original = proc.ar2dit

    def _recording_ar2dit(source_outputs, prompts=None, _requires_multimodal_data=False):
        result = original(source_outputs, prompts, _requires_multimodal_data)
        for dit_prompt in result:
            info = dit_prompt["additional_information"]
            hidden = info["full_hidden_states"]
            sink.append(
                {
                    "full_token_ids": list(info["full_token_ids"]),
                    "answer_start_index": int(info["answer_start_index"][0]),
                    "hidden_states": hidden.detach().to(device="cpu", dtype=torch.float32).clone(),
                }
            )
        return result

    proc.ar2dit = _recording_ar2dit
    try:
        yield
    finally:
        proc.ar2dit = original


def _tp_deploy_config(tp_size: int) -> str:
    """Materialize a deploy yaml with stage 0 at ``tp_size``.

    Stage 1 (DiT) is left on device 0 at TP=1: DiT-side parallelism is out of
    scope, and holding it constant keeps every downstream difference
    attributable to the AR stage.  Stage 0's ``gpu_memory_utilization`` of 0.5
    and stage 1's 0.3 still share device 0, exactly as the shipped config does.
    """
    return modify_stage_config(
        BASE_DEPLOY_CONFIG,
        updates={
            "stages": {
                0: {
                    "tensor_parallel_size": tp_size,
                    "devices": ",".join(str(i) for i in range(tp_size)),
                },
                1: {
                    "tensor_parallel_size": 1,
                    "devices": "0",
                },
            }
        },
    )


def _build_prompt(ar_width: int, ar_height: int, gen_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the t2i prompt through the production helper.

    Reusing ``build_text_to_image_prompt`` means the harness cannot drift from
    the prompt the serving layer actually sends.  The DiT knobs and the visual
    placeholder ids it does not set are added on top.
    """
    prompt = build_text_to_image_prompt(PROMPT_TEXT, None, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    info = prompt["additional_information"]
    assert info["ar_width"] == [ar_width] and info["ar_height"] == [ar_height], (
        f"build_text_to_image_prompt derived a different grid: {info['ar_width']}x{info['ar_height']} "
        f"vs expected {ar_width}x{ar_height}"
    )
    # Cross-check the module constants against the checkpoint: the structural
    # layer below asserts against these ids, so a stale constant would validate
    # the wrong columns.
    for key, cfg_key in (
        ("eol_token_id", "eol_token_id"),
        ("visual_token_start_id", "visual_token_start_id"),
        ("visual_token_end_id", "visual_token_end_id"),
    ):
        assert info[key][0] == int(gen_cfg[cfg_key]), (
            f"model_extras {key}={info[key][0]} disagrees with the checkpoint's "
            f"t2i_generation_config.json ({gen_cfg[cfg_key]})"
        )
    info.update(
        {
            "num_inference_steps": [DIT_INFERENCE_STEPS],
            "text_guidance_scale": [DIT_GUIDANCE_SCALE],
            "cfg_range": DIT_CFG_RANGE,
            "visual_ids": [
                _IMAGE_TOKEN_ID,
                _VIDEO_TOKEN_ID,
                _VISION_START_TOKEN_ID,
                _VISION_END_TOKEN_ID,
            ],
        }
    )
    return prompt


def _sample_pixels(img_tensor: torch.Tensor) -> tuple[float, ...]:
    """Sample fixed fractional coordinates from a (C, H, W) or (1, C, H, W) tensor."""
    t = img_tensor.float().clamp(0.0, 1.0)
    if t.ndim == 4:
        t = t[0]
    _, height, width = t.shape
    return tuple(
        round(float(t[c, min(int(rh * (height - 1)), height - 1), min(int(rw * (width - 1)), width - 1)]), 6)
        for c, rh, rw in _PIXEL_SAMPLE_COORDS
    )


def _extract_image_pixels(outputs: list[Any]) -> tuple[float, ...] | None:
    for out in outputs:
        for request_output in out if isinstance(out, list) else [out]:
            for completion in getattr(request_output, "outputs", None) or []:
                mm = getattr(completion, "multimodal_output", None)
                if not (isinstance(mm, dict) and "image" in mm):
                    continue
                images = mm["image"] if isinstance(mm["image"], list) else [mm["image"]]
                for img in images:
                    if isinstance(img, torch.Tensor) and img.ndim in (3, 4):
                        return _sample_pixels(img)
    return None


def _run_capture(tp_size: int) -> ARCapture:
    """Start one engine at ``tp_size``, run the fixed prompt, tear it down."""
    ar_height, ar_width = IMAGE_HEIGHT // AR_PATCH_SIZE, IMAGE_WIDTH // AR_PATCH_SIZE
    # ar_height rows of ar_width visual tokens, each row closed by one eol.
    grid_tokens = ar_height * (ar_width + 1)

    prompt = _build_prompt(ar_width, ar_height, _load_t2i_gen_config())
    ar_sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=grid_tokens + 1, detokenize=False)
    dit_sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    sink: list[dict[str, Any]] = []
    try:
        with _record_stage_output(sink):
            with OmniRunner(
                get_model_prefix() + MODEL_PATH,
                seed=SEED,
                deploy_config=_tp_deploy_config(tp_size),
            ) as runner:
                outputs = list(runner.omni.generate([prompt], [ar_sampling, dit_sampling]))
    finally:
        # Leave no engine behind: the next TP degree reuses the same devices.
        gc.collect()
        cleanup_test_environment()

    assert sink, (
        f"TP={tp_size}: the AR stage never reached ar2dit, so no stage output was produced. "
        "Either the AR stage failed or the pipeline did not advance to the DiT stage."
    )
    assert len(sink) == 1, f"TP={tp_size}: expected one AR->DiT handoff, recorded {len(sink)}"
    recorded = sink[0]
    answer_start = recorded["answer_start_index"]
    full_token_ids = tuple(int(t) for t in recorded["full_token_ids"])

    return ARCapture(
        tp_size=tp_size,
        generated_token_ids=full_token_ids[answer_start:],
        full_token_ids=full_token_ids,
        answer_start_index=answer_start,
        hidden_states=recorded["hidden_states"],
        image_pixels=_extract_image_pixels(outputs),
    )


# One engine per TP degree for the whole module.  Failures are cached too, so a
# degree that cannot start does not get retried once per check layer.
_CAPTURES: dict[int, ARCapture | BaseException] = {}


def capture(tp_size: int) -> ARCapture:
    if tp_size not in _CAPTURES:
        try:
            _CAPTURES[tp_size] = _run_capture(tp_size)
        except BaseException as exc:  # noqa: BLE001 - re-raised immediately
            _CAPTURES[tp_size] = exc
            raise
    cached = _CAPTURES[tp_size]
    if isinstance(cached, BaseException):
        pytest.fail(f"TP={tp_size} capture failed earlier in this module: {cached!r}")
    return cached


def _model_dir() -> Path:
    """Resolve the checkpoint directory.

    ``MODEL_PREFIX`` points at a local mirror in some CI environments, in which
    case there is nothing to download; fall back to the hub otherwise.
    """
    prefix = get_model_prefix()
    if prefix:
        local = Path(prefix + MODEL_PATH)
        if local.exists():
            return local
    return Path(snapshot_download(MODEL_PATH))


def _load_t2i_gen_config() -> dict[str, Any]:
    cfg_path = _model_dir() / "t2i_generation_config.json"
    if not cfg_path.exists():
        pytest.skip(f"t2i_generation_config.json not found at {cfg_path}")
    with cfg_path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# L0 - vocabulary sharding, config only, no accelerator
# ---------------------------------------------------------------------------
@pytest.mark.cpu
@pytest.mark.parametrize("tp_size", TP_DEGREES)
def test_gen_vocab_padding_and_sharding(tp_size: int) -> None:
    """Pin down how the extra generation vocabulary pads and shards.

    ``gen_vocab_size = 32800`` is *not* a multiple of the 64-element vocab
    padding unit, so the padded width (32832) differs from the logical width.
    ``compute_logits`` concatenates the base and gen logits, and
    ``_apply_t2i_token_constraints`` then indexes the result by absolute token
    id -- which is only correct if each half has had its padding sliced off
    first, i.e. if the concatenation offset equals ``base_vocab_size`` at every
    TP degree.

    This runs without an accelerator, so it gives CI signal on the hypothesis
    before any of the multi-GPU layers below can run.
    """
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            DEFAULT_VOCAB_PADDING_SIZE,
            pad_vocab_size,
        )
    except ImportError as exc:  # pragma: no cover - vLLM layout change
        pytest.skip(f"vLLM vocab-parallel helpers unavailable: {exc}")

    # The Preview AR text config, constructed from its own defaults: the
    # top-level Mammothmoda2Config leaves ``llm_config`` unset when built with
    # no arguments, so it carries no text config to read.
    from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Qwen2_5_VLTextConfig

    text_config = Mammothmoda2Qwen2_5_VLTextConfig()
    gen_vocab_size = int(text_config.gen_vocab_size)
    base_vocab_size = int(text_config.gen_vocab_start_index)
    total_vocab_size = int(text_config.vocab_size)

    assert total_vocab_size == base_vocab_size + gen_vocab_size, (
        "compute_logits emits base+gen logits and the sampler requires the last dimension to equal "
        f"model_config.get_vocab_size(); config says {total_vocab_size} != {base_vocab_size} + {gen_vocab_size}"
    )

    padded_gen = pad_vocab_size(gen_vocab_size, DEFAULT_VOCAB_PADDING_SIZE)
    padded_base = pad_vocab_size(base_vocab_size, DEFAULT_VOCAB_PADDING_SIZE)

    # The premise of this whole test file.  If a future config makes 32800 an
    # exact multiple, the padding-offset hypothesis stops applying and the
    # comment above should be revisited rather than the assertion relaxed.
    assert padded_gen != gen_vocab_size, (
        f"gen_vocab_size={gen_vocab_size} now pads to itself ({padded_gen}); the padding-offset failure "
        "mode this file was written for no longer applies -- re-derive the risk before deleting coverage."
    )

    for name, padded in (("base", padded_base), ("gen", padded_gen)):
        assert padded % tp_size == 0, (
            f"TP={tp_size}: padded {name} vocab {padded} is not divisible by the TP degree, so the "
            "per-rank shard widths are unequal and the gathered logits cannot be sliced uniformly"
        )

    # What compute_logits relies on: each half is sliced back to its logical
    # width before torch.cat, so the gen half starts exactly at base_vocab_size.
    assert base_vocab_size + gen_vocab_size == total_vocab_size
    assert padded_base >= base_vocab_size and padded_gen >= gen_vocab_size

    print(
        f"\n[tp{tp_size}] vocab: base={base_vocab_size} (padded {padded_base}, "
        f"shard {padded_base // tp_size}), gen={gen_vocab_size} (padded {padded_gen}, "
        f"shard {padded_gen // tp_size}), concat offset must be {base_vocab_size}"
    )


# ---------------------------------------------------------------------------
# L0.5 - grid structure, single TP degree, no reference run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp_size", TP_PARAMS)
def test_ar_grid_structure(tp_size: int) -> None:
    """The emitted AR grid must be well-formed at this TP degree on its own.

    ``_apply_t2i_token_constraints`` forces column ``ar_width`` of every row to
    be ``eol`` and restricts every other column to the visual-token range.  The
    constraint indexes the concatenated logits by absolute token id, so a
    shifted concat offset shows up here as an out-of-range token -- without
    needing a TP=1 reference to compare against.

    This is the layer to look at first when a degree goes red.
    """
    cap = capture(tp_size)
    gen_cfg = _load_t2i_gen_config()
    eol = int(gen_cfg["eol_token_id"])
    visual_start = int(gen_cfg["visual_token_start_id"])
    visual_end = int(gen_cfg["visual_token_end_id"])
    ar_width = IMAGE_WIDTH // AR_PATCH_SIZE
    ar_height = IMAGE_HEIGHT // AR_PATCH_SIZE

    tokens = cap.generated_token_ids
    grid_tokens = ar_height * (ar_width + 1)
    # The AR emits the full grid plus one trailing token, and ``ar2dit`` drops
    # that last one (it has no hidden state), so the capture is exactly the
    # grid.  recipes/MammothModa2/MammothModa2.md records 4,161 generated
    # tokens at 1024x1024, i.e. 64 rows x (64 visual + 1 eol) + 1 = grid + 1.
    assert len(tokens) == grid_tokens, (
        f"TP={tp_size}: captured {len(tokens)} grid tokens, expected {grid_tokens} "
        f"({ar_height} rows x ({ar_width} visual + 1 eol)). A short capture means the AR stage "
        "stopped early -- resolve that before reading the position checks below."
    )

    violations: list[str] = []
    for i, token in enumerate(tokens):
        row, column = divmod(i, ar_width + 1)
        if column == ar_width:
            if token != eol:
                violations.append(f"index {i} (row {row}, end-of-row): expected eol={eol}, got {token}")
        elif not visual_start <= token <= visual_end:
            violations.append(
                f"index {i} (row {row}, column {column}): expected a visual token in "
                f"[{visual_start}, {visual_end}], got {token}"
            )
        if len(violations) >= 8:
            violations.append("... further violations suppressed")
            break

    assert not violations, (
        f"TP={tp_size}: the t2i token constraints did not hold. This is the signature of a shifted "
        "base/gen logits concatenation offset -- check compute_logits and the padding of gen_head.\n  "
        + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# L1 - token parity against TP=1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp_size", COMPARISON_TP_PARAMS)
def test_ar_token_parity(tp_size: int) -> None:
    """Greedy decoding must select the same tokens at every TP degree.

    #7114 commits to identical output at a fixed seed.  TP reorders the
    all-reduce summations, so logits differ in the last bits; with
    ``temperature=0, top_k=1`` that only changes the selected token where two
    candidates are near-tied.  If this fails, the failure message says whether
    the divergence looks like a near-tie (a numerics finding to report) or a
    structural jump (a layout bug to fix).
    """
    reference = capture(REFERENCE_TP)
    cap = capture(tp_size)

    assert len(cap.generated_token_ids) == len(reference.generated_token_ids), (
        f"TP={tp_size} generated {len(cap.generated_token_ids)} tokens, "
        f"TP={REFERENCE_TP} generated {len(reference.generated_token_ids)}"
    )

    first_divergence = next(
        (i for i, (a, b) in enumerate(zip(cap.generated_token_ids, reference.generated_token_ids)) if a != b),
        None,
    )
    if first_divergence is None:
        return

    ar_width = IMAGE_WIDTH // AR_PATCH_SIZE
    row, column = divmod(first_divergence, ar_width + 1)
    got = cap.generated_token_ids[first_divergence]
    want = reference.generated_token_ids[first_divergence]
    matching = sum(1 for a, b in zip(cap.generated_token_ids, reference.generated_token_ids) if a == b)
    pytest.fail(
        f"TP={tp_size}: token divergence from the TP={REFERENCE_TP} reference at index {first_divergence} "
        f"(row {row}, column {column} of {ar_width}): got {got}, expected {want}. "
        f"{matching}/{len(reference.generated_token_ids)} tokens matched overall.\n"
        f"  |got - expected| = {abs(got - want)}. A small delta between two neighbouring visual tokens "
        "points at a near-tie under reordered all-reduce (report it in the TP findings); a large jump, "
        "or a token outside the visual range, points at a logits-layout bug."
    )


# ---------------------------------------------------------------------------
# L2a - stage output invariants, single TP degree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp_size", TP_PARAMS)
def test_ar_stage_output_alignment(tp_size: int) -> None:
    """The AR->DiT hidden states must be complete and token-aligned.

    ``ar2dit`` asserts only ``hidden_total == len(prompt) + len(generated)``.
    That length can be right while the contents are not: under TP the hidden
    states are replicated by the row-parallel all-reduce, and only the driver
    rank's copy reaches the orchestrator.  These checks cover what the length
    assertion cannot.
    """
    cap = capture(tp_size)
    hidden = cap.hidden_states

    assert hidden.ndim == 2, f"TP={tp_size}: expected (num_tokens, hidden_size), got shape {tuple(hidden.shape)}"
    assert hidden.shape[0] == len(cap.full_token_ids), (
        f"TP={tp_size}: {hidden.shape[0]} hidden states for {len(cap.full_token_ids)} tokens -- "
        "the stage output is not token-aligned"
    )
    assert cap.answer_start_index > 0, (
        f"TP={tp_size}: answer_start_index={cap.answer_start_index} leaves no prompt span"
    )
    assert cap.answer_start_index < hidden.shape[0], (
        f"TP={tp_size}: answer_start_index={cap.answer_start_index} is past the end of the "
        f"{hidden.shape[0]}-token stage output"
    )
    assert torch.isfinite(hidden).all(), f"TP={tp_size}: stage output contains NaN or Inf"

    # A rank that contributed nothing shows up as an all-zero span; check the
    # prompt and generated halves separately so a partially gathered tensor
    # cannot hide behind a healthy prompt prefix.
    prompt_span = hidden[: cap.answer_start_index]
    generated_span = hidden[cap.answer_start_index :]
    for name, span in (("prompt", prompt_span), ("generated", generated_span)):
        assert span.numel() > 0, f"TP={tp_size}: the {name} span of the stage output is empty"
        dead_rows = int((span.abs().sum(dim=-1) == 0).sum())
        assert dead_rows == 0, (
            f"TP={tp_size}: {dead_rows}/{span.shape[0]} rows of the {name} span are all-zero -- "
            "the stage output looks partially gathered"
        )

    print(
        f"\n[tp{tp_size}] stage output: shape={tuple(hidden.shape)} dtype={hidden.dtype} "
        f"answer_start={cap.answer_start_index} abs_mean={float(hidden.abs().mean()):.6f}"
    )


# ---------------------------------------------------------------------------
# L2b - stage output parity against TP=1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp_size", COMPARISON_TP_PARAMS)
def test_ar_stage_output_parity(tp_size: int) -> None:
    """AR->DiT hidden states must agree with the TP=1 reference.

    These are what conditions the DiT, so a drift here changes the image even
    when every token id matched.  Tolerances are loose on purpose (see
    ``HIDDEN_ATOL``); the measured numbers are printed either way so PR-3 can
    report per-degree drift instead of only pass/fail.
    """
    reference = capture(REFERENCE_TP)
    cap = capture(tp_size)

    assert cap.hidden_states.shape == reference.hidden_states.shape, (
        f"TP={tp_size}: stage output shape {tuple(cap.hidden_states.shape)} != "
        f"TP={REFERENCE_TP} shape {tuple(reference.hidden_states.shape)}"
    )
    assert cap.answer_start_index == reference.answer_start_index, (
        f"TP={tp_size}: answer_start_index {cap.answer_start_index} != "
        f"TP={REFERENCE_TP} {reference.answer_start_index} -- the prompt/completion boundary moved"
    )

    got, want = cap.hidden_states, reference.hidden_states
    abs_diff = (got - want).abs()
    max_abs = float(abs_diff.max())
    max_rel = float((abs_diff / want.abs().clamp_min(1e-6)).max())
    cosine = float(torch.nn.functional.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1), dim=-1).item())
    worst_token = int(abs_diff.max(dim=-1).values.argmax())
    print(
        f"\n[tp{tp_size}] stage-output drift vs tp{REFERENCE_TP}: "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} cosine={cosine:.8f} worst_token={worst_token}"
    )

    assert cosine >= HIDDEN_MIN_COSINE, (
        f"TP={tp_size}: stage output cosine similarity {cosine:.8f} < {HIDDEN_MIN_COSINE}. "
        "This is well beyond reordered-all-reduce noise -- the AR stage is producing different "
        f"conditioning, not just a differently rounded one. Worst token index: {worst_token}."
    )
    assert torch.allclose(got, want, atol=HIDDEN_ATOL, rtol=HIDDEN_RTOL), (
        f"TP={tp_size}: stage output exceeds tolerance (max_abs={max_abs:.6e} > atol={HIDDEN_ATOL}, "
        f"max_rel={max_rel:.6e} > rtol={HIDDEN_RTOL}) at token {worst_token}, "
        f"though cosine similarity ({cosine:.8f}) stayed above {HIDDEN_MIN_COSINE}. "
        "Localized drift like this usually means one token's routing differed -- check gen_token_mask."
    )


# ---------------------------------------------------------------------------
# L3 - end-to-end pixel backstop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp_size", COMPARISON_TP_PARAMS)
def test_t2i_image_parity(tp_size: int) -> None:
    """End-to-end backstop: same prompt and seed, same image.

    The DiT runs at TP=1 for every degree, so if the layers above are green
    this one is redundant by construction -- which is the point.  It failing
    while L1/L2 pass means something outside the AR stage output differs, and
    it passing on its own proves nothing about layout: an image is produced
    either way.  Never treat this as the primary signal.
    """
    reference = capture(REFERENCE_TP)
    cap = capture(tp_size)

    if reference.image_pixels is None:
        pytest.skip(f"TP={REFERENCE_TP} produced no image tensor; nothing to compare against")
    assert cap.image_pixels is not None, f"TP={tp_size}: pipeline produced no image tensor"

    mismatches = [
        f"pixel {i}: got {got:.6f}, expected {want:.6f} (delta {abs(got - want):.2e})"
        for i, (got, want) in enumerate(zip(cap.image_pixels, reference.image_pixels))
        if abs(got - want) > PIXEL_ATOL
    ]
    assert not mismatches, (
        f"TP={tp_size}: {len(mismatches)}/{len(reference.image_pixels)} sampled pixels differ from the "
        f"TP={REFERENCE_TP} reference by more than {PIXEL_ATOL}.\n  " + "\n  ".join(mismatches)
    )
