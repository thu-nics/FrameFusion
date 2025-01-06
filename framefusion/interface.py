from types import MethodType

# model types
from transformers import  LlavaNextVideoForConditionalGeneration
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

# replace methods
from framefusion.models.llava_next_video.modeling_llava_next_video import _merge_input_ids_with_image_features_get_token_type
from framefusion.models.llava_video.modeling_llava_video import prepare_inputs_labels_for_multimodal_get_token_type
from framefusion.models.qwen2.modeling_qwen2 import replace_minicpmv_merge_then_fastv_cost_given, replace_framefusion_forward
from framefusion.models.minicpmv.modeling_minicpmv import get_vllm_embedding


def get_token_type(model):
    # LlavaNextVideo Model
    if isinstance(model, LlavaNextVideoForConditionalGeneration):
        model._merge_input_ids_with_image_features = MethodType(
            _merge_input_ids_with_image_features_get_token_type, model
        )

    # LlavaVideo Model
    elif isinstance(model, LlavaQwenForCausalLM):
        model.prepare_inputs_labels_for_multimodal = MethodType(
            prepare_inputs_labels_for_multimodal_get_token_type, model
        )

    # MiniCPM Model
    elif model.config.architectures[0] == "MiniCPMV":
        model.get_vllm_embedding = MethodType(get_vllm_embedding, model)
    else:
        raise NotImplementedError


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
        model._merge_input_ids_with_image_features = MethodType(
            _merge_input_ids_with_image_features_get_token_type, model
        )
        replace_framefusion_forward(
            model,
            cost=cost,
            similarity_lower_bound=similarity_lower_bound,
            ratio_lower_bound=ratio_lower_bound,
            llm_key="model",
            decoder_layer_key="model.layers",
        )

    # LlavaVideo Model
    elif isinstance(model, LlavaQwenForCausalLM):
        model.prepare_inputs_labels_for_multimodal = MethodType(
            prepare_inputs_labels_for_multimodal_get_token_type, model
        )
        replace_framefusion_forward(
            model,
            cost=cost,
            similarity_lower_bound=similarity_lower_bound,
            ratio_lower_bound=ratio_lower_bound,
            llm_key="model",
            decoder_layer_key="model.layers",
        )

    # MiniCPM Model
    elif model.config.architectures[0] == "MiniCPMV":
        model.get_vllm_embedding = MethodType(get_vllm_embedding, model)
        replace_framefusion_forward(
            model,
            cost=cost,
            similarity_lower_bound=similarity_lower_bound,
            ratio_lower_bound=ratio_lower_bound,
            llm_key="llm.model",
            decoder_layer_key="llm.model.layers",
        )
    else:
        raise NotImplementedError
