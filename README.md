# FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models

**[[arXiv](https://arxiv.org/abs/2501.01986)]** **[[project page](https://thu-nics.github.io/FrameFusion_Project_Page/)]**

FrameFusion reduces the number of tokens in Large Vision-Language Models (LVLMs) by combining similarity-based merging with importance-based pruning. It achieves a 70% vision token reduction, 3.4–4.4× LLM speedups, and 1.6–1.9× end-to-end speedups with minimal performance impact.

Feel free to star the repo or cite the paper if you find it interesting.

```bibtex
@misc{fu2024framefusion,
      title={FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models}, 
      author={Tianyu Fu and Tengxuan Liu and Qinghao Han and Guohao Dai and Shengen Yan and Huazhong Yang and Xuefei Ning and Yu Wang},
      year={2024},
      eprint={2501.01986},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01986}, 
}
```

## News

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

### Working with other models

`NVILA` and `Llava-Video` have conflicting architecture definitions. FrameFusion is compatible with both, but please install only one of them to avoid conflict between the two repos.

#### Llava-Video

To use Llava-Video LVLM, you also need to install the dependencies for it. We recommend clone the [official repository](https://github.com/LLaVA-VL/LLaVA-NeXT), then install it with `pip install -e .` in the cloned repository.

#### NVILA

1. Clone the [VILA](https://github.com/NVlabs/VILA) repo.

2. Run `./environment_setup.sh` to install NVILA dependencies in the current conda environment.

3. Install VILA with `pip install -e .` in the cloned repository.


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


