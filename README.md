# FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models

**[[arXiv](https://arxiv.org/abs/2501.01986)]** **[[project page](https://thu-nics.github.io/FrameFusion_Project_Page/)]**

FrameFusion reduces the number of tokens in Large Vision-Language Models (LVLMs) by combining similarity-based merging with importance-based pruning. It achieves a 70% vision token reduction, 3.4–4.4× LLM speedups, and 1.6–1.9× end-to-end speedups with minimal performance impact.

Feel free to star the repo or cite the paper if you find it interesting.

```bibtex
@article{fu2024framefusion,
  title={FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models},
  author={Fu, Tianyu and Liu, Tengxuan and Han, Qinghao and Dai, Guohao and Yan, Shengen and Yang, Huazhong and Ning, Xuefei and Wang, Yu},
  journal={arXiv preprint arXiv:2501.01986},
  year={2024}
}
```

## News

* [2025/05] Support Qwen2-VL and InternVL2.5

* [2025/04] Support NVILA model family


## Environment Setup

### General
Create a new environment:

```bash
conda create -n framefusion python=3.10
conda activate framefusion
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install FrameFusion:

```bash
pip install -e .
```

### Working with Other Models

**Important:** `NVILA` and `Llava-Video` have conflicting architectures. **FrameFusion** supports both, but please install only one to avoid conflicts.

#### Option 1: Llava-Video
To install Llava-Video LVLM dependencies:

1. Clone the [LLaVA-NeXT repository](https://github.com/LLaVA-VL/LLaVA-NeXT):
   ```bash
   git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
   cd LLaVA-NeXT
   ```
2. Install via:
   ```bash
   pip install -e .
   ```

#### Option 2: NVILA
To install NVILA dependencies:

1. Clone the [VILA repository](https://github.com/NVlabs/VILA):
   ```bash
   git clone https://github.com/NVlabs/VILA.git
   cd VILA
   ```
2. Run environment setup script to install dependencies in current conda environment:
   ```bash
   ./environment_setup.sh
   ```
3. Install via:
   ```bash
   pip install -e .
   ```


## How to

### Run an example

We provide an example with LLaVA-Video-7B model to inference on a video with or without FrameFusion in `script/playground/example_llava.py`.

```bash
python script/playground/example_llava.py
```

### Apply FrameFusion

You can apply FrameFusion in your own code to any huggingface model that supports the interface with few lines of code. Here is an example:

```python
from llava.model.builder import load_pretrained_model
from framefusion.interface import apply_framefusion

# set attn_implementation to be sdpa
tokenizer, model, image_processor, max_length = load_pretrained_model("lmms-lab/LLaVA-Video-7B-Qwen2", None, "llava_qwen", torch_dtype="bfloat16", attn_implementation='sdpa', device_map="auto")

# apply FrameFusion
apply_framefusion(model, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1)

# use the model as usual
```

### Adept to new models

#### Understand Code Structure

- `framefusion/`: The main package for FrameFusion.
    - `models/`: The adapter for different models.
    - `main.py`: The main implementation of FrameFusion.
    - `interface.py`: The interface for applying FrameFusion.
- `scripts/`: Scripts for running experiments.
    - `evaluate/`: Scripts for evaluating the performance models.
    - `playground/`: Scripts for running misc experiments.
- `example/`: Example input videos

#### Modify the code

1. Add a new model adapter in `framefusion/models/`, it applies framefusion after the attention module. 

    > Three model functions are required: `llm_forward`, `decoder_forward`, and `attention_forward`. The forward functions are easily modified from the corresponding `modeling_<MODEL>.py` functions in huggingface transformers. All modifications are marked with `###` comments. For LLM, see `framefusion/models/qwen2/modeling_qwen2.py` as an example.

2. Register the model in `framefusion/interface.py`, it applies framefusion to the correct model class.

3. Add a new example in `script/playground/`, it shows how to apply framefusion to the model.

#### Happy to help

If you have any questions on applying FrameFusion to a new model, please feel free to open an issue. We are happy to help you and expand the adapter for more models.

## Supported Model List

### MimiCPM-V

* [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)

### Llava-Video

* [lmms-lab/LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)
* [lmms-lab/LLaVA-Video-72B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2)



### NVILA

* [Efficient-Large-Model/NVILA-Lite-2B](https://huggingface.co/Efficient-Large-Model/NVILA-Lite-2B)
* [Efficient-Large-Model/NVILA-8B-Video](https://huggingface.co/Efficient-Large-Model/NVILA-8B-Video)
* [Efficient-Large-Model/NVILA-Lite-15B-Video](https://huggingface.co/Efficient-Large-Model/NVILA-Lite-15B-Video)

### Qwen2-VL

* [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

### InternVL2_5

* [OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B)


