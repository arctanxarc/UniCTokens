import ast
import json
import logging
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def remove_token(raw_str):
    pattern = r"<token_\d+>"
    return re.sub(pattern, "", raw_str)


def get_image_files(directory):
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = []
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and any(file_name.lower().endswith(ext.lower()) for ext in image_extensions):
            image_files.append(file_path)
    return image_files


def check_embedding_dtype(model, input_ids, target_dtype):
    embed_layer = model.showo.get_input_embeddings()
    embed_output = embed_layer(input_ids)
    assert embed_output.dtype == target_dtype, (
        f"Embedding output dtype mismatch: expected {target_dtype}, got {embed_output.dtype}"
    )
    logging.debug("Embedding dtype check passed")


def check_dtype(original_model, target_dtype):
    for name, param in original_model.named_parameters():
        if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"):
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                logging.warning("Parameter %s cast to %s", name, target_dtype)
    for _, buf in original_model.named_buffers():
        if buf.dtype.is_floating_point and buf.dtype != target_dtype:
            buf.data = buf.data.to(target_dtype)
    return original_model


def mkdir(path):
    folder_path = Path(path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=False)
    elif not folder_path.is_dir():
        os.remove(folder_path)
        folder_path.mkdir(parents=True, exist_ok=False)


def read_json_to_dict(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: invalid JSON file: {file_path}")
        return None
    except Exception as exc:
        print(f"Error reading file {file_path}: {exc}")
        return None


def save_distributed_model(model, optimizer, save_dir, epoch=0):
    os.makedirs(save_dir, exist_ok=True)
    save_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    save_path = os.path.join(save_dir, f"model_epoch{epoch}.pt")
    torch.save(save_data, save_path)
    print(f"[save] {save_path}")


def load_distributed_model(model, optimizer=None, save_dir=None, device="cuda"):
    if not save_dir or not os.path.exists(save_dir):
        raise FileNotFoundError(f"Model save directory does not exist: {save_dir}")

    model_files = [
        file_path
        for file_path in Path(save_dir).iterdir()
        if file_path.is_file() and file_path.name.startswith("model_epoch") and file_path.suffix == ".pt"
    ]
    if not model_files:
        raise FileNotFoundError(f"No model_epoch*.pt file found in {save_dir}")

    latest_model_file = max(model_files, key=lambda x: int(x.stem.replace("model_epoch", "")))
    load_data = torch.load(latest_model_file, map_location=device, weights_only=True)
    model.load_state_dict(load_data["model_state_dict"])

    if optimizer and load_data.get("optimizer_state_dict"):
        optimizer.load_state_dict(load_data["optimizer_state_dict"])

    epoch = load_data["epoch"]
    model.to(device)
    return model, optimizer, epoch


def load_single_model_weights_from_file(model, weight_file_path, device="cuda"):
    if not os.path.exists(weight_file_path):
        raise FileNotFoundError(f"Model weight file does not exist: {weight_file_path}")

    load_data = torch.load(weight_file_path, map_location=device, weights_only=True)
    required_keys = ["model_state_dict", "epoch"]
    for key in required_keys:
        if key not in load_data:
            raise KeyError(f"Weight file {weight_file_path} is missing required key: {key}")

    model.load_state_dict(load_data["model_state_dict"])
    model.to(device)
    return model, load_data["epoch"]


def calculate_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Two lists must have the same length")
    squared_diff_sum = 0.0
    for a, b in zip(list1, list2):
        squared_diff_sum += (a - b) ** 2
    return squared_diff_sum


def calculate_bleu(reference, candidate):
    reference_tokens = list(reference)
    candidate_tokens = list(candidate)
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, weights=(0.7, 0.3), smoothing_function=smoothie)


def extract_list_from_response(response_text: str) -> List[Any]:
    bracket_match = re.search(r"\[[\s\S]*\]", response_text)
    if bracket_match:
        candidate = bracket_match.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
                if isinstance(value, list):
                    return value
            except Exception:
                pass

    numbered_items = re.findall(r"(?:\d+[\.)]|-|\*)\s*([^\n]+)", response_text)
    if numbered_items:
        return [item.strip() for item in numbered_items]

    line_items = []
    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped and stripped not in {"```", "```json", "```python"}:
            line_items.append(stripped)
    return line_items


def extract_and_clean_list(response_text: str) -> List[str]:
    raw_list = extract_list_from_response(response_text)
    cleaned_list = []
    for item in raw_list:
        if isinstance(item, str):
            item = re.sub(r"^[\d\-\.\*\)\s]+", "", item.strip())
            item = re.sub(r"[\.!,:;]+$", "", item)
            if item:
                cleaned_list.append(item)
        else:
            cleaned_list.append(item)
    return cleaned_list


def get_image_path(path):
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = []
    for file_name in os.listdir(path):
        file_path = Path(path) / file_name
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions and file_path.stem.isdigit():
            image_files.append((int(file_path.stem), file_path))

    if not image_files:
        return 0, None, None

    image_files.sort(key=lambda x: x[0])
    selected_num, selected_path = random.choice(image_files)
    return len(image_files), str(selected_path), selected_num


def get_questions(n):
    questions_file = os.environ.get("SYNC_R1_QUESTIONS_FILE")
    if not questions_file:
        raise RuntimeError("SYNC_R1_QUESTIONS_FILE is not set")
    with open(questions_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    selected_dialog = None
    for name, items in data.items():
        if name != "text_only" and str(n) in name:
            selected_dialog = random.choice(items)
            break
    text_only = random.choice(data.get("text_only", [])) if data.get("text_only") else None
    return selected_dialog, text_only


def normalize_logits(logits, eps=1e-8):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + eps)


def extract_single_number(text):
    match = re.fullmatch(r"\s*-?\d+\.?\d*\s*", text.strip())
    if match:
        num_str = match.group().strip()
        return float(num_str) if "." in num_str else int(num_str)
    return 0.5


def manage_top_images(image_path, score, folder_path, top_n=30, counter_file="counter.txt"):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    counter_path = os.path.join(folder_path, counter_file)
    if os.path.exists(counter_path):
        try:
            with open(counter_path, "r", encoding="utf-8") as file:
                counter = int(file.read().strip())
        except Exception:
            counter = 0
    else:
        counter = 0

    score_json_path = os.path.join(folder_path, "score.json")
    if os.path.exists(score_json_path):
        try:
            with open(score_json_path, "r", encoding="utf-8") as file:
                score_dict = json.load(file)
        except Exception:
            score_dict = {}
    else:
        score_dict = {}

    current_images = len(score_dict)
    should_add = current_images < top_n
    if not should_add and score_dict:
        should_add = score > min(score_dict.values())

    if should_add:
        if current_images >= top_n and score_dict:
            min_score = min(score_dict.values())
            for img_name, img_score in list(score_dict.items()):
                if img_score == min_score:
                    img_path = os.path.join(folder_path, img_name)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    del score_dict[img_name]
                    break

        counter += 1
        new_img_name = f"image_{counter:06d}.png"
        new_img_path = os.path.join(folder_path, new_img_name)
        shutil.copy2(image_path, new_img_path)
        score_dict[new_img_name] = score

        with open(counter_path, "w", encoding="utf-8") as file:
            file.write(str(counter))
        with open(score_json_path, "w", encoding="utf-8") as file:
            json.dump(score_dict, file, indent=4)

    return should_add


def face_recognition_score(generated_image_path, data_root, concept):
    import face_recognition

    generated_image = face_recognition.load_image_file(generated_image_path)
    generated_encodings = face_recognition.face_encodings(generated_image)
    if not generated_encodings:
        raise RuntimeError("face_recognition failed to detect a face in the generated image")
    generated_encoding = generated_encodings[0]

    path = os.path.join(data_root, "concept/train", concept)
    paths = get_image_files(path)
    scores = []
    for image_path in paths:
        try:
            known_image = face_recognition.load_image_file(image_path)
            known_encodings = face_recognition.face_encodings(known_image)
            if not known_encodings:
                continue
            distance = face_recognition.face_distance([known_encodings[0]], generated_encoding)[0]
        except Exception:
            continue

        max_distance = 1.0
        if distance >= max_distance:
            similarity = 0.0
        else:
            similarity = 1 - distance / max_distance
            similarity = 0 if similarity < 0.5 else (similarity - 0.5) * 2
        scores.append(similarity)

    if scores:
        return sum(scores) / len(scores)
    raise RuntimeError("face_recognition scoring failed")


_FACENET_DEVICE = None
identity_detector = None
mtcnn = None
identity_model_file = "./facenet_20180402_114759_vggface2.pth"


def _get_facenet_device():
    configured_device = os.environ.get("FACENET_DEVICE")
    if configured_device:
        return configured_device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _ensure_facenet_runtime():
    global _FACENET_DEVICE, identity_detector, mtcnn

    if identity_detector is not None and mtcnn is not None:
        return _FACENET_DEVICE, identity_detector, mtcnn

    _FACENET_DEVICE = _get_facenet_device()
    identity_detector = InceptionResnetV1(
        pretrained=None,
        classify=False,
        num_classes=None,
        dropout_prob=0.6,
        device=_FACENET_DEVICE,
    )
    identity_detector.logits = nn.Linear(512, 8631)
    identity_detector.load_state_dict(torch.load(identity_model_file, map_location=_FACENET_DEVICE))
    identity_detector.eval()
    identity_detector.to(_FACENET_DEVICE)

    mtcnn = MTCNN(
        image_size=160,
        margin=32,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=_FACENET_DEVICE,
    )
    return _FACENET_DEVICE, identity_detector, mtcnn


def facenet_score(image_path1, data_root, concept):
    device, identity_detector, mtcnn = _ensure_facenet_runtime()
    img1 = Image.open(image_path1).convert("RGB")
    face_tensor1 = mtcnn(img1)
    if face_tensor1 is None:
        raise RuntimeError("facenet failed to detect a face in the generated image")

    with torch.no_grad():
        identity_embeddings1 = identity_detector(face_tensor1.unsqueeze(0).to(device))

    path = os.path.join(data_root, "concept/train", concept)
    paths = get_image_files(path)
    scores = []
    for image_path in paths:
        try:
            img2 = Image.open(image_path).convert("RGB")
            face_tensor2 = mtcnn(img2)
            if face_tensor2 is None:
                continue
            with torch.no_grad():
                identity_embeddings2 = identity_detector(face_tensor2.unsqueeze(0).to(device))
            cos_sim = torch.nn.functional.cosine_similarity(identity_embeddings1, identity_embeddings2)
            cos_value = float(cos_sim.item())
            scores.append(0 if cos_value < 0.5 else 2 * (cos_value - 0.5))
        except Exception:
            continue

    if scores:
        return sum(scores) / len(scores)
    raise RuntimeError("facenet scoring failed")
