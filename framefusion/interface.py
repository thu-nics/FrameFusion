# common imports
from types import MethodType
from typing import Callable
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module
from transformers import PreTrainedModel

# framefusion methods
from framefusion.main import FrameFusion
from framefusion.utils import TEXT_TOKEN, IGNORE_TOKEN, get_attr_by_name

# model types
from transformers import LlavaNextVideoForConditionalGeneration
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

# replace methods
from framefusion.models.llava_next_video.modeling_llava_next_video import _merge_input_ids_with_image_features_get_token_type
from framefusion.models.llava_video.modeling_llava_video import prepare_inputs_labels_for_multimodal_get_patch_type
from framefusion.models.minicpmv.modeling_minicpmv import get_vllm_embedding
from framefusion.models.qwen2.modeling_qwen2 import Qwen2Model_merge_then_fastv_cost_given_forward, Qwen2DecoderLayer_merge_then_prune_by_cost_forward, Qwen2SdpaAttention_merge_then_prune_by_cost_forward


def apply_framefusion(model, cost, similarity_lower_bound, ratio_lower_bound):
    """
    Apply FrameFusion to the model

    Args:
        model: the model to apply FrameFusion to
        cost: the cost of the FrameFusion
        similarity_lower_bound: the similarity lower bound of the FrameFusion
        ratio_lower_bound: the ratio lower bound of the FrameFusion
    """
    # LlavaNextVideo Model
    if isinstance(model, LlavaNextVideoForConditionalGeneration):
        model._merge_input_ids_with_image_features = MethodType(_merge_input_ids_with_image_features_get_token_type, model)

        llm_forward = Qwen2Model_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2DecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2SdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"

    # LlavaVideo Model
    elif isinstance(model, LlavaQwenForCausalLM):
        model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type, model)

        llm_forward = Qwen2Model_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2DecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2SdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"

    # MiniCPM Model
    elif model.config.architectures[0] == "MiniCPMV":

        model.get_vllm_embedding = MethodType(get_vllm_embedding, model)
        llm_forward = Qwen2Model_merge_then_fastv_cost_given_forward
        decoder_forward = Qwen2DecoderLayer_merge_then_prune_by_cost_forward
        attention_forward = Qwen2SdpaAttention_merge_then_prune_by_cost_forward
        llm_key = "llm.model"
        decoder_key = "layers"
        attention_key = "self_attn"

    else:
        raise NotImplementedError

    replace_framefusion_forward(
        model,
        cost=cost,
        similarity_lower_bound=similarity_lower_bound,
        ratio_lower_bound=ratio_lower_bound,
        llm_forward=llm_forward,
        decoder_forward=decoder_forward,
        attention_forward=attention_forward,
        llm_key=llm_key,
        decoder_key=decoder_key,
        attention_key=attention_key,
    )


def get_token_type(model):
    # LlavaNextVideo Model
    if isinstance(model, LlavaNextVideoForConditionalGeneration):
        model._merge_input_ids_with_image_features = MethodType(_merge_input_ids_with_image_features_get_token_type, model)

    # LlavaVideo Model
    elif isinstance(model, LlavaQwenForCausalLM):
        model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type, model)

    # MiniCPM Model
    elif model.config.architectures[0] == "MiniCPMV":
        model.get_vllm_embedding = MethodType(get_vllm_embedding, model)
    else:
        raise NotImplementedError


def replace_framefusion_forward(
    module: torch.nn.Module,
    cost: float,
    similarity_lower_bound: float,
    ratio_lower_bound: float,
    llm_forward: Callable,
    decoder_forward: Callable,
    attention_forward: Callable,
    llm_key: str = "model",
    decoder_key: str = "layers",
    attention_key: str = "self_attn",
):
    """
    Replace the forward method of the model with the framefusion forward method.
    Make framefusion a property of the model.

    The keys are accessed in an hierarchical manner: llm_key -> decoder_key -> attention_key. Each key can have multiple hierarchies, e.g. "llm.model", which will be accessed by module.llm.model
    """
    framefusion = FrameFusion(cost, similarity_lower_bound, ratio_lower_bound)

    module.framefusion = framefusion

    llm = get_attr_by_name(module, llm_key)
    assert isinstance(llm, PreTrainedModel), f"{llm_key} is not a PreTrainedModel"

    llm.framefusion = framefusion
    llm.forward = MethodType(llm_forward, llm)

    decoder_layers = get_attr_by_name(llm, decoder_key)
    for i, decoder_layer in enumerate(decoder_layers):
        assert isinstance(decoder_layer, nn.Module), f"{decoder_key}[{i}] is not a nn.Module"

        decoder_layer.framefusion = framefusion
        decoder_layer.forward = MethodType(decoder_forward, decoder_layer)

        # ensure accelerate hooks are not removed
        if hasattr(decoder_layer, "_hf_hook"):
            decoder_layer._old_forward = MethodType(decoder_forward, decoder_layer)
            add_hook_to_module(decoder_layer, decoder_layer._hf_hook)

        qwen2_attention_instance = get_attr_by_name(decoder_layer, attention_key)
        assert isinstance(qwen2_attention_instance, nn.Module), f"{decoder_key}[{i}].self_attn is not a nn.Module"

        # replace the forward method of the attention layer
        qwen2_attention_instance.framefusion = framefusion
        qwen2_attention_instance.forward = MethodType(attention_forward, qwen2_attention_instance)
