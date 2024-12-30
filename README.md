# FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models

FrameFusion reduces the number of tokens in Large Vision-Language Models (LVLMs) by combining similarity-based merging with importance-based pruning. It achieves a 70% vision token reduction, 3.4–4.4× LLM speedups, and 1.6–1.9× end-to-end speedups with minimal performance impact.

## Environment Setup

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

To use Llava-Video LVLM, you also need to install the dependencies for it. We recommend clone the [official repository](https://github.com/LLaVA-VL/LLaVA-NeXT), then install it with `pip install -e .` in the cloned repository.

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
    - `main.py`: The main implementation of FrameFusion.
    - `models/`: The adapter for different models.
    - `interface.py`: The interface for applying FrameFusion.
- `scripts/`: Scripts for running experiments.
    - `evaluate/`: Scripts for evaluating the performance models.
    - `playground/`: Scripts for running misc experiments.
- `example/`: Example input videos

#### Modify the code

1. Add a new model adapter in `framefusion/models/`, it applies framefusion after the attention module.
2. Register the model in `framefusion/interface.py`, it applies framefusion to the correct model class.
3. Add a new example in `script/playground/`, it shows how to apply framefusion to the model.
