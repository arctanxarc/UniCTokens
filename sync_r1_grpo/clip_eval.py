from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

try:
    import clip
except ModuleNotFoundError:
    local_clip_root = Path(__file__).resolve().parent / "CLIP-main"
    if str(local_clip_root) not in sys.path:
        sys.path.insert(0, str(local_clip_root))
    import clip


def _as_path_list(paths: Sequence[str | Path] | str | Path) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def load_train_image_paths(data_root: str | Path, concept: str) -> list[Path]:
    data_root = Path(data_root)
    train_images_file = data_root / "concept" / "train" / "train_images.json"
    ref_images_dir = data_root / "concept" / "train" / concept

    with train_images_file.open("r", encoding="utf-8") as handle:
        train_images = json.load(handle)

    return [ref_images_dir / image_name for image_name in train_images[concept]]


@dataclass
class ClipScore:
    clip_i: float
    clip_t: float | None = None


class CLIPEvaluator:
    """Reusable CLIP scorer for image-image and text-image similarity."""

    def __init__(
        self,
        device: str | torch.device,
        clip_model_path: str | Path,
        dtype: torch.dtype | None = None,
    ) -> None:
        clip_model_path = Path(clip_model_path).expanduser().resolve()
        os.environ.setdefault("CLIP_CACHE_DIR", str(clip_model_path.parent))
        requested_device = torch.device(device)
        self.device = requested_device
        self._warned_cpu_fallback = False

        try:
            model, preprocess = clip.load(str(clip_model_path), device=self.device)
        except RuntimeError as exc:
            if requested_device.type != "cuda":
                raise
            self.device = torch.device("cpu")
            self._warn_cpu_fallback(exc)
            model, preprocess = clip.load(str(clip_model_path), device=self.device)

        self.model = model.eval()
        if dtype is not None and self.device.type != "cpu":
            self.model = self.model.to(dtype=dtype)
        self.preprocess = preprocess

    def _warn_cpu_fallback(self, error: RuntimeError) -> None:
        if self._warned_cpu_fallback:
            return
        print(
            "[warn] CLIP scorer could not initialize on CUDA and is falling back to CPU. "
            f"Original error: {error}",
            flush=True,
        )
        self._warned_cpu_fallback = True

    def _encode_images(self, image_paths: Sequence[str | Path] | str | Path) -> torch.Tensor:
        path_list = _as_path_list(image_paths)
        images = []
        for image_path in path_list:
            image = Image.open(image_path).convert("RGB")
            images.append(self.preprocess(image))
        batch = torch.stack(images, dim=0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
        return features / features.norm(dim=-1, keepdim=True)

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def image_to_image_similarity(
        self,
        reference_images: Sequence[str | Path] | str | Path,
        generated_images: Sequence[str | Path] | str | Path,
    ) -> float:
        ref_features = self._encode_images(reference_images)
        gen_features = self._encode_images(generated_images)
        similarity = ref_features @ gen_features.T
        return float(similarity.mean().item())

    def text_to_image_similarity(
        self,
        text: str,
        generated_images: Sequence[str | Path] | str | Path,
    ) -> float:
        text_features = self._encode_text(text)
        gen_features = self._encode_images(generated_images)
        similarity = text_features @ gen_features.T
        return float(similarity.mean().item())


class SHOWOConceptClipEvaluator:
    """Concept-level wrapper around CLIP scoring for the Show-o evaluation setup."""

    def __init__(
        self,
        device: str | torch.device,
        clip_model_path: str | Path,
        data_root: str | Path,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.clip = CLIPEvaluator(
            device=device,
            clip_model_path=clip_model_path,
            dtype=dtype,
        )

    def reference_paths(self, concept: str) -> list[Path]:
        return load_train_image_paths(self.data_root, concept)

    def score_prompt_images(
        self,
        concept: str,
        image_paths: Sequence[str | Path] | str | Path,
        prompt: str | None = None,
        reference_image_paths: Sequence[str | Path] | str | Path | None = None,
    ) -> ClipScore:
        ref_paths = _as_path_list(reference_image_paths) if reference_image_paths else self.reference_paths(concept)
        clip_i = self.clip.image_to_image_similarity(ref_paths, image_paths)
        clip_t = None
        if prompt:
            clip_t = self.clip.text_to_image_similarity(prompt, image_paths)
        return ClipScore(clip_i=clip_i, clip_t=clip_t)


__all__ = [
    "CLIPEvaluator",
    "ClipScore",
    "SHOWOConceptClipEvaluator",
    "load_train_image_paths",
]
