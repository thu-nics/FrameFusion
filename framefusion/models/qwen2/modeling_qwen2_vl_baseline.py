from types import MethodType
from functools import partial
import math
from transformers import Qwen2VLForConditionalGeneration
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.models.qwen2_vl.modeling_qwen2_vl import QWEN2_VL_INPUTS_DOCSTRING, Qwen2VLCausalLMOutputWithPast, _CONFIG_FOR_DOC, Qwen2VLModel, Qwen2VLDecoderLayer, Qwen2VLSdpaAttention, apply_multimodal_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, SinkCache
from transformers.utils import logging
import torch
from typing import Optional, List, Tuple, Union
from torch.nn import CrossEntropyLoss
logger = logging.get_logger(__name__)
from framefusion.utils import scaled_dot_product_attention
from minference import streaming_forward


TEXT_TOKEN = -1


def replace_qwenvl_forward(model, mode="merge_then_fastv_cost_given", **kwargs):
    model.mode = mode
    print(f"replace_qwenvl_forward mode: {mode} and kwargs: {kwargs}")
    if mode=="fastv":
        fastv_kwargs = {
            "fastv_k": kwargs.get("fastv_k", 3),
            "fastv_r": kwargs.get("fastv_r", 0.5)
        }
        print(f"Config\n{fastv_kwargs}")

        replace_qwenvl_fastv(
            model,
            **fastv_kwargs
        )
    elif mode=="streamingllm":
        streamingllm_kwargs = {
            "init_num": kwargs.get("init_num", 8),
            "length_rate": kwargs.get("length_rate", 0.3),
        }
        print(f"Config\n{streamingllm_kwargs}")

        replace_qwenvl_streamingllm(
            model,
            **streamingllm_kwargs
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented yet.")


def replace_qwenvl_streamingllm(model, init_num = 4, length_rate = 0.3):
    model.init_num = init_num
    model.length_rate = length_rate

    if isinstance(model.model, Qwen2VLModel):
        model.model.forward = MethodType(partial(Qwen2VLModel_streamingllm_forward, model=model), model.model)
    for i, decoder_layer in enumerate(model.model.layers):
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2VLSdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2VLSdpaAttention_streamingllm_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")


def replace_qwenvl_fastv(model, fastv_k = 3, fastv_r = 0.5):
    model.fastv_k = fastv_k
    model.fastv_r = fastv_r 
    
    if isinstance(model.model, Qwen2VLModel):
        model.model.forward = MethodType(partial(Qwen2VLModel_fastv_forward, model=model), model.model)
    for i, decoder_layer in enumerate(model.model.layers):
        if isinstance(decoder_layer, Qwen2VLDecoderLayer):
            decoder_layer.forward=MethodType(Qwen2VLDecoderLayer_fastv_forward, decoder_layer)
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2VLSdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2VLSdpaAttention_fastv_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")



def Qwen2VLModel_fastv_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    model = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    ### change position_embeddings into a list for future pruning
    position_embeddings = list(position_embeddings) 

    ### end changing

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    ### implement fastv
    FASTV_k = model.fastv_k # the layer_idx to prune
    FASTV_r = model.fastv_r # the pruning ratio
    FASTV_image_token_start_index = model.image_token_start_index.item()
    FASTV_image_token_length = model.image_token_length
    device = self.device

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

                
        if use_cache:
            if hidden_states.shape[1] != 1:
                if layer_idx<FASTV_k:
                    pruned_attention_mask = causal_mask

                elif layer_idx==FASTV_k:
                    # compute pruned tokens, generate fastv sign
                    last_layer_attention = layer_outputs[1]
                    # compute average attention over different head
                    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                    # generate new attention mask based on the average attention, sample the top ATTENTION_RANK tokens with highest attention
                    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                    # get the attention in image token
                    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[FASTV_image_token_start_index:FASTV_image_token_start_index+FASTV_image_token_length]
                    # get the indexs of the top ATTENTION_RANK tokens
                    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(round(FASTV_image_token_length*(1-FASTV_r))).indices + FASTV_image_token_start_index
                    # keep index
                    keep_indexs = torch.cat( (torch.arange(FASTV_image_token_start_index,device=device), top_attention_rank_index, torch.arange(FASTV_image_token_start_index+FASTV_image_token_length,hidden_states.shape[1],device=device)))
                    # sort index
                    keep_indexs = keep_indexs.sort().values
                    # update seq length
                    new_seq_length = keep_indexs.shape[0]
                    # filter hidden states
                    hidden_states = hidden_states[:,keep_indexs,:] 
                    # update position ids
                    position_ids = keep_indexs.unsqueeze(0)
                    # update position embeddings
                    position_embeddings[0] = position_embeddings[0][:,:,keep_indexs,:]
                    position_embeddings[1] = position_embeddings[1][:,:,keep_indexs,:]

                    cache_position = cache_position[:new_seq_length]
            else:
                pruned_attention_mask = causal_mask
        
        else:
            raise NotImplementedError("fastv only support use_cache=True")
        

        if layer_idx == FASTV_k - 1:
            output_attentions = True
        else:
            output_attentions = False
    
        layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=pruned_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
        )


        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

    ### force output_attentions to be False(we store attn_weights by ourselves)
        output_attentions = False
    ### end
    
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def Qwen2VLDecoderLayer_fastv_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    ### adding attn_weights if needed
    if self_attn_weights != None:
        outputs += (self_attn_weights,)
    ### end adding


    if use_cache:
        outputs += (present_key_value,)

    return outputs


def Qwen2VLSdpaAttention_fastv_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    model = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    ### storing attnetion weights if needed
    attn_weights = None

    if self.layer_idx != model.fastv_k - 1:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        last_query = query_states[:,:,-1,:].view(bsz, self.num_heads, 1, self.head_dim)    
        scale_factor = 1 / math.sqrt(query_states.size(-1))
        attn_weights = last_query@ key_states.transpose(-2, -1) * scale_factor 
        attn_weights = torch.softmax(attn_weights, dim=-1)
            
        ### finish storing attention weights

    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    #     query_states,
    #     key_states,
    #     value_states,
    #     attn_mask=causal_mask,
    #     dropout_p=self.attention_dropout if self.training else 0.0,
    #     is_causal=is_causal,
    # )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def Qwen2VLModel_streamingllm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    model = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
    
    init_num = model.init_num
    if inputs_embeds != None:
        local_window_num = int(inputs_embeds.shape[1] * model.length_rate) - init_num

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = SinkCache(window_length = init_num + local_window_num, num_sink_tokens= init_num )
        # past_key_values = DynamicCache()

    ### changing past_key_values into SinkCache     
    if len(past_key_values.key_cache) == 0:
        past_key_values = SinkCache(window_length = init_num + local_window_num, num_sink_tokens= init_num )
    ### end changing

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



# Adapted from Qwen2Attention.forward
def Qwen2VLSdpaAttention_streamingllm_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    model = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    ### start implement StreamingLLM
    init_num = model.init_num
    local_window_num = int(model.length_rate * hidden_states.shape[1]) - init_num
    if q_len == 1:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
    else:
        attn_output = streaming_forward(query_states, key_states, value_states, init_num, local_window_num)
    ### end implementing streamingLLM

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value