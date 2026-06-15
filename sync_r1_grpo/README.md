# Uni-Synergy

This folder contains the cleaned Sync-R1 GRPO training path on top of the UniCTokens codebase.

It is not a fully standalone project by itself. The code in this folder still imports core modules from the repository root, including `models/`, `training/`, and `llava/`. The goal of this README is to help a new user start from zero and reach a runnable GRPO training setup.

## What This Folder Contains

- `train_grpo.py`: GRPO training entrypoint
- `grpo.py`: trajectory-level MaskGIT GRPO implementation
- `ref_model.py`: reference-side helper
- `clip_eval.py`, `glm_api.py`, `pdata.py`, `utils.py`: runtime helpers
- `configs/showo_demo.yaml`
- `configs/showo_demo_512x512.yaml`
- `requirements.txt`

## Recommended Platform

The current launcher initializes distributed training with `backend='nccl'`, so the recommended environment is:

- Linux or WSL2 with NVIDIA GPU support
- Python 3.10
- CUDA-capable PyTorch

Native Windows is not the main target environment for this code path. In particular:

- `torch.distributed` with `nccl` is usually easier on Linux or WSL2
- `face-recognition` can be harder to install on native Windows

## 1. Clone The Full UniCTokens Repository

This folder depends on the root repository structure, so clone the whole repo instead of only copying `sync_r1_grpo/`.

```bash
git clone https://github.com/arctanxarc/UniCTokens.git
cd UniCTokens
```

## 2. Create The Environment

Using `conda`:

```bash
conda create -n sync-r1 python=3.10 -y
conda activate sync-r1
pip install --upgrade pip
pip install -r sync_r1_grpo/requirements.txt
```

If `clip @ git+https://github.com/openai/CLIP.git` cannot be installed directly, install it manually first, then rerun `pip install -r sync_r1_grpo/requirements.txt`.

## 3. Download Or Prepare The UniCTokens Dataset

Dataset links from the main UniCTokens release:

- Google Drive: [UniCTokens dataset](https://drive.google.com/file/d/1bRv_E855P2ds6_1YeyQtJ7kfUxntPoGa/view?usp=sharing)
- Hugging Face: [HankYang428/unictokens_data](https://huggingface.co/datasets/HankYang428/unictokens_data)

After download, make sure your dataset root looks like:

```text
unictokens_data/
|-- black_512x512.png
|-- concepts_list.json
|-- concept/
|   |-- train/
|   |   `-- <concept_name>/
|   `-- test/
|       `-- <concept_name>/
`-- ...
```

The training command will use this path as:

```text
--data_root path/to/unictokens_data
```

If you downloaded only the raw dataset package and still need derived JSON files, follow the dataset generation instructions in the repository root README and run the upstream data generation scripts first.

## 4. Download The Base Model Weights

This GRPO stage expects local model directories that can be loaded by `from_pretrained()`.

You need local paths for:

- `MAGVITv2`
- `Show-o` or `Show-o-512x512`
- `Phi-1.5`

Edit:

- `sync_r1_grpo/configs/showo_demo.yaml`
- `sync_r1_grpo/configs/showo_demo_512x512.yaml`

and replace the placeholders:

```yaml
model:
  vq_model:
    vq_model_name: "path/to/magvitv2"

  showo:
    pretrained_model_path: "path/to/show-o-512x512"
    llm_model_path: "path/to/phi-1_5"
```

Notes:

- Use local directories or local Hugging Face snapshot paths, not just model names.
- `ref_model.py` uses `local_files_only=True`, so these assets must already exist on disk.

## 5. Download The Extra Reward Weights

Besides the main base models, this training path also expects two local reward-related weight files:

1. CLIP weight:

```text
sync_r1_grpo/ViT-B-32.pt
```

2. FaceNet weight:

```text
sync_r1_grpo/facenet_20180402_114759_vggface2.pth
```

The current code looks for them relative to `sync_r1_grpo/`, so keeping them directly inside this folder is the simplest choice.

On first run, InsightFace may also download its `antelopev2` assets automatically. If you run in an offline environment, prepare those assets locally in advance.

## 6. Prepare The Personalized Initialization Checkpoint

This is the most important dependency for Sync-R1.

`train_grpo.py` does not start from the raw base model alone. It expects a concept-specific initialization checkpoint containing:

- `epoch_<N>_embed.pt`
- `epoch_<N>_lm_head_weight.pt`
- `epoch_<N>_lm_head_bias.pt`

### Where Do These Files Come From?

They normally come from the earlier UniCTokens personalized training stages in this same repository.

Typical upstream preparation flow:

```bash
concept="bo"
data_root="path/to/unictokens_data"

python train_w_3_stages/train_p_stage_1.py \
  --concept "${concept}" \
  --data_root "${data_root}" \
  --task_name test_train_s1 \
  --need_new_tokens \
  --mmu_data \
  --init_by_images \
  --need_init

python train_w_3_stages/train_p_stage_2.py \
  --concept "${concept}" \
  --data_root "${data_root}" \
  --task_name test_train_s2 \
  --pre_trained_ckpt_name test_train_s1 \
  --t2i_data \
  --mmu_data
```

Those upstream scripts save checkpoints under:

```text
saves/<concept>/<task_name>/
```

For example:

```text
saves/bo/test_train_s2/
|-- epoch_1_embed.pt
|-- epoch_1_lm_head_weight.pt
|-- epoch_1_lm_head_bias.pt
|-- ...
`-- epoch_15_*.pt
```

### Important Path Convention In `sync_r1_grpo`

The current Sync-R1 launcher loads checkpoints from:

```text
../<concept>/<pre_trained_ckpt_name>/
```

when you run it from inside `sync_r1_grpo/`.

So the easiest way to match the current code without editing it is to copy or link the stage-2 checkpoint folder to:

```text
<repo_root>/<concept>/<pre_trained_ckpt_name>/
```

Example target layout:

```text
UniCTokens/
|-- sync_r1_grpo/
|-- bo/
|   `-- test_train_s2/
|       |-- epoch_15_embed.pt
|       |-- epoch_15_lm_head_weight.pt
|       `-- epoch_15_lm_head_bias.pt
`-- ...
```

For example, from the repository root:

```powershell
New-Item -ItemType Directory -Force -Path .\bo | Out-Null
Copy-Item -Recurse .\saves\bo\test_train_s2 .\bo\
```

Then use:

```text
--concept bo
--pre_trained_ckpt_name test_train_s2
--epoch_to_load 15
```

## 7. Quick Start Choice: LLM Reward Backends

To quickly verify the training process, you can use LLM-based reward helpers first. If you want the full scoring path, configure environment variables before launching:

- `ZAI_API_KEY`
- `VERTEXAI_PROJECT`
- `VERTEXAI_LOCATION`
- `GOOGLE_APPLICATION_CREDENTIALS`

Examples:

```powershell
$env:ZAI_API_KEY="your_zhipu_key"
$env:VERTEXAI_PROJECT="your_gcp_project"
$env:VERTEXAI_LOCATION="us-central1"
$env:GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

We are currently resolving configuration issues regarding the deployment of the reward expert for real-world use. We will release the corresponding code as soon as possible once it is fixed.

## 8. Launch Training

Because this folder imports repository-root modules and also uses checkpoint paths relative to `sync_r1_grpo/`, the most reliable way to launch is:

1. `cd` into `sync_r1_grpo/`
2. set `PYTHONPATH=..`
3. run `torchrun`

### 3 GPUs

PowerShell:

```powershell
cd .\sync_r1_grpo
$env:PYTHONPATH='..'
torchrun --nproc_per_node=3 train_grpo.py `
  --num_gpus 3 `
  --config_file configs/showo_demo_512x512.yaml `
  --data_root path/to/unictokens_data `
  --pre_trained_ckpt_name test_train_s2 `
  --concept bo `
  --save_dir ./tmp_result_accelerate/ `
  --epoch_to_load 15 `
  --batch_num 10 `
  --batch_size 1 `
  --num_gen 9 `
  --llm glm `
  --accelerate True `
  --semantic True
```

### 1 GPU

```powershell
cd .\sync_r1_grpo
$env:PYTHONPATH='..'
torchrun --nproc_per_node=1 train_grpo.py `
  --num_gpus 1 `
  --config_file configs/showo_demo_512x512.yaml `
  --data_root path/to/unictokens_data `
  --pre_trained_ckpt_name test_train_s2 `
  --concept bo `
  --save_dir ./tmp_result_accelerate/ `
  --epoch_to_load 15 `
  --batch_num 10 `
  --batch_size 1 `
  --num_gen 3 `
  --llm glm `
  --accelerate True `
  --semantic True
```

## 9. Runtime Assumptions

The current code path assumes:

- `batch_size=1`
- one prompt is expanded into multiple rollouts with `--num_gen`
- `num_gen` should be divisible by `num_gpus`

Useful arguments:

- `--save_dir`: where logs, generated images, and saved model weights go
- `--epoch_to_load`: which personalized initialization epoch to load
- `--llm`: currently `glm` or `gemini`
- `--semantic`: enables extra semantic reward shaping

## 10. Output Structure

Training outputs are written under `--save_dir`, typically including:

- `logs/<concept>/`
- `images/<concept>/Epoch*/`
- `model_weights/<concept>/`
- `accelerate/<concept>/`

## 11. Common Failure Cases

### `ProcessGroupNCCL` or distributed init errors

Use Linux or WSL2 with CUDA. The launcher currently assumes `torchrun` plus NCCL.

### Missing `epoch_*_embed.pt` files

Check that:

- you trained or obtained the concept checkpoint first
- the files exist for the chosen `--epoch_to_load`
- the folder is placed at `../<concept>/<pre_trained_ckpt_name>/` relative to `sync_r1_grpo/`

### Missing `ViT-B-32.pt` or FaceNet weights

Put:

- `ViT-B-32.pt`
- `facenet_20180402_114759_vggface2.pth`

directly in `sync_r1_grpo/`.

### Path placeholders still unchanged

Double-check:

- `sync_r1_grpo/configs/showo_demo.yaml`
- `sync_r1_grpo/configs/showo_demo_512x512.yaml`
- `--data_root`
- `--pre_trained_ckpt_name`

## Minimal Checklist Before First Run

- Full UniCTokens repo cloned
- Python environment created
- `pip install -r sync_r1_grpo/requirements.txt` completed
- dataset downloaded and `--data_root` points to it
- base model directories downloaded and config placeholders replaced
- `sync_r1_grpo/ViT-B-32.pt` exists
- `sync_r1_grpo/facenet_20180402_114759_vggface2.pth` exists
- concept checkpoint copied to `<repo_root>/<concept>/<ckpt_name>/`
- launch executed from inside `sync_r1_grpo/`
