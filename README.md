# UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens

## UniCTokens Dataset

> Download the dataset here: https://drive.google.com/file/d/1R933C8ko0p41HJks_5B6me41eS1WR3aE/view?usp=drive_link

### Data Overview

| Item                   | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| **Total concepts**     | 20 (Human × 10 · Animal × 5 · Object × 5)                                              |
| **Images per concept** | **N ≈ 10 – 15** (already split into *train* / *test*)                                  |
| **Negative samples**   | `random_images/` (100 random irrelevant images) + `negative_example/` (hard negatives) |


### Benchmark Tasks

#### MMU (Multi-Modal Understanding)

| Sub-task         | Source files                      | Evaluation focus                                                 |
| ---------------- | --------------------------------- | ---------------------------------------------------------------- |
| **Text-Only QA** | `test/<concept>/text_only.json`   | Check whether the model remembers concept knowledge (no image) |
| **VQA**          | `test/<concept>/vqa.json` + image | Visual question answering about the concept image                |
| **Rec**          | `test/*.png`            | Pure visual recognition capability                               |

#### T2I (Text-to-Image Generation)

| Mode                              | Input                                                           | Metrics                                                           |
| --------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Vanilla generation**            | Prompts from the DreamBooth Dataset → target-concept images | CLIP-I / CLIP-T · ArcFace similarity                              |
| **Personalized knowledge-driven** | `t2i_conditions.json`                                           | Combined T2I-Score: must satisfy both visual & textual attributes |


### Directory Structure

```text
UniCTokens/
├── black_512x512.png          # Pure black placeholder
├── concepts_list.json         # List of 20 concept names
├── template.json              # Template for generating training data
├── random_images/             # 100 simple negative samples for training
│   ├── 0.png
│   └── … 99.png
├── concept/                   # 🔑 Concept data (train / test)
│   ├── train/
│   │   └── <concept_name>/    # 20 folders
│   │       ├── 0.png … N.png          # Original training images
│   │       ├── cropped/               # Cropped regions
│   │       ├── info.json              # Concept profile & extra info
│   │       ├── conversations.json     # Training dialogues
│   │       ├── positive_recognitions.json  # Positive QA pairs
│   │       ├── random_recognitions.json    # Negative QA pairs
│   │       └── negative_example/      # Hard negatives + score.json
│   └── test/
│       └── <concept_name>/
│           ├── 0.png … 4.png
│           ├── text_only.json         # Text-only QA
│           ├── vqa.json               # VQA pairs
│           └── t2i_conditions.json    # Conditions for knowledge-driven T2I
├── gen_showo_training_data.py   # Script to create Stage-1/2/3 training files
├── gen_test_data.py             # Script to create all evaluation files
└── README.md
```


### Quick Start For Dataset

1. **Set the dataset root**
   Open `gen_showo_training_data.py` and `gen_test_data.py`, change

   ```python
   DATA_ROOT = "/path/to/UniCTokens_Dataset"
   ```

   to the actual dataset path.

2. **Generate data**

   ```bash
   # Create Stage-1/2/3 training samples
   python gen_showo_training_data.py

   # Create MMU & T2I evaluation samples
   python gen_test_data.py
   ```


## License

The dataset and code are released under **CC-BY-NC 4.0** and are intended for academic research **only**. Commercial use is not permitted.


## Citation

```bibtex
@article{an2025unictokens,
  title={UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens},
  author={An, Ruichuan and Yang, Sihan and Zhang, Renrui and Shen, Zijun and Lu, Ming and Dai, Gaole and Liang, Hao and Guo, Ziyu and Yan, Shilin and Luo, Yulin and others},
  journal={arXiv preprint arXiv:2505.14671},
  year={2025}
}
```


## Contact

* GitHub Issues: [https://github.com/arctanxarc/UniCTokens/issues](https://github.com/arctanxarc/UniCTokens/issues)
* Email: [arctanxarc@gmail.com](mailto:arctanxarc@gmail.com)

