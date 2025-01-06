from types import MethodType
from functools import partial
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from transformers.cache_utils import Cache, DynamicCache,DynamicCache, SinkCache
from transformers.models.qwen2.modeling_qwen2 import repeat_kv,apply_rotary_pos_emb, logger, QWEN2_INPUTS_DOCSTRING
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.doc import add_start_docstrings_to_model_forward
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention, Qwen2DecoderLayer, Qwen2Model
from functools import partial
try:
    from minference import streaming_forward
except ImportError:
    # minference is not needed if streamingllm is not used
    streaming_forward = None

from framefusion.utils import TEXT_TOKEN, IGNORE_TOKEN
from framefusion.main import find_contigious_latter_index

"""
Utils
"""

def compute_density_overhead(sparsity_list) -> tuple:
    """
    Compute the average cumulative product and total product of the sparsity list.
    """
    density_list = [1-s for s in sparsity_list]

    cost = 0.0
    remaining_density = 1.0 
    for density in density_list:
        remaining_density *= density
        cost += remaining_density 

    norm_cost = cost / len(density_list)
    return norm_cost, remaining_density

"""
Meta Interface
"""

def replace_Qwen2_forward(model, mode="merge_then_fastv_cost_given", **kwargs):
    print(f"replace_Qwen2_forward mode: {mode} and kwargs: {kwargs}")
    
    if mode=="prefill_merge":
        prefill_merge_kwargs = {
            "sparsity": kwargs.get("sparsity", [0.0] * 28),
        }
        
        print(f"Config\n{prefill_merge_kwargs}")

        cost, remaining_density=compute_density_overhead(prefill_merge_kwargs['sparsity']) 
        print(f"Computational cost: {cost:.3f}, Remaining density: {remaining_density:.3f}")

        replace_Qwen2_merging(
            model,
            **prefill_merge_kwargs
        )
    elif mode=="fastv":
        fastv_kwargs = {
            "fastv_k": kwargs.get("fastv_k", 3),
            "fastv_r": kwargs.get("fastv_r", 0.5)
        }
        print(f"Config\n{fastv_kwargs}")

        replace_Qwen2_fastv(
            model,
            **fastv_kwargs
        )
    elif mode=="merge_then_fastv":
        merge_then_fastv_kwargs = {
            "sparsity": kwargs.get("sparsity", [0.1] * 28),
            "fastv_k": kwargs.get("fastv_k", 3),
            "fastv_r": kwargs.get("fastv_r", 0.5)
        }
        print(f"Config\n{merge_then_fastv_kwargs}")

        replace_Qwen2_merge_then_fastv(
            model,
            **merge_then_fastv_kwargs
        )
    elif mode=="streamingllm":
        streamingllm_kwargs = {
            "init_num": kwargs.get("init_num", 8),
            "length_rate": kwargs.get("length_rate", 0.3),
        }
        print(f"Config\n{streamingllm_kwargs}")

        replace_Qwen2_streamingllm(
            model,
            **streamingllm_kwargs
        )
    elif mode=="fastv_then_merge":
        fastv_then_merge_kwargs = {
            "fastv_k": kwargs.get("fastv_k", 2),
            "fastv_r": kwargs.get("fastv_r", 0.75),
            "merging_sparsity": kwargs.get("merging_sparsity", 0.3)
        }
        print(f"Config\n{fastv_then_merge_kwargs}")

        replace_Qwen2_fastv_then_merge(
            model,
            **fastv_then_merge_kwargs
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented yet.")
    
def replace_minicpmv_forward(model, mode="fastv", **kwargs):
    print(f"replace_minicpmv_forward mode: {mode} and kwargs: {kwargs}")
    if mode=="fastv":
        fastv_kwargs = {
            "fastv_k": kwargs.get("fastv_k", 3),
            "fastv_r": kwargs.get("fastv_r", 0.5)
        }
        print(f"Config\n{fastv_kwargs}")

        replace_minicpmv_fastv(
            model,
            **fastv_kwargs
        )
    elif mode=="streamingllm":
        streamingllm_kwargs = {
            "init_num": kwargs.get("init_num", 8),
            "length_rate": kwargs.get("length_rate", 0.3),
        }
        print(f"Config\n{streamingllm_kwargs}")

        replace_minicpmv_streamingllm(
            model,
            **streamingllm_kwargs
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented yet.")


"""
Forward functions
"""

"""
Fastv forward functions
"""

def replace_Qwen2_fastv(model, fastv_k = 3, fastv_r = 0.5):
    model.fastv_k = fastv_k
    model.fastv_r = fastv_r 
    
    if isinstance(model.model, Qwen2Model):
        model.model.forward = MethodType(partial(Qwen2Model_fastv_forward, model=model), model.model)
    for i, decoder_layer in enumerate(model.model.layers):
        if isinstance(decoder_layer, Qwen2DecoderLayer):
            decoder_layer.forward=MethodType(Qwen2DecoderLayer_fastv_forward, decoder_layer)
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2SdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2SdpaAttention_fastv_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")
     
def replace_minicpmv_fastv(model, fastv_k = 3, fastv_r = 0.5):
    model.fastv_k = fastv_k
    model.fastv_r = fastv_r 
    
    if isinstance(model.llm.model, Qwen2Model):
        model.llm.model.forward = MethodType(partial(Qwen2Model_fastv_forward, model=model), model.llm.model)
    for i, decoder_layer in enumerate(model.llm.model.layers):
        if isinstance(decoder_layer, Qwen2DecoderLayer):
            decoder_layer.forward=MethodType(Qwen2DecoderLayer_fastv_forward, decoder_layer)
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2SdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2SdpaAttention_fastv_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")

@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2Model_fastv_forward(
    self,
    input_ids: torch.LongTensor = None,
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
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

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
    FASTV_image_token_length = model.image_token_length.item()
    device = self.device
    #seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1] (here because cache position in minicpmv is not none,so past_seen_tokens is not defined )
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # pruning hidden states, no kv cache
        
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
                    position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                    position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]

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
    ### end fastv

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
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def Qwen2SdpaAttention_fastv_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        model = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.to(query_states.device), sin.to(query_states.device))

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

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

def Qwen2DecoderLayer_fastv_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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


"""
StreamingLLM forward functions
"""
def replace_Qwen2_streamingllm(model, init_num = 4, length_rate = 0.3):
    model.init_num = init_num
    model.length_rate = length_rate

    if isinstance(model.model, Qwen2Model):
        model.model.forward = MethodType(partial(Qwen2Model_streamingllm_forward, model=model), model.model)
    for i, decoder_layer in enumerate(model.model.layers):
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2SdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2SdpaAttention_streamingllm_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")

def replace_minicpmv_streamingllm(model, init_num = 4, length_rate = 0.3):
    model.init_num = init_num
    model.length_rate = length_rate

    if isinstance(model.llm.model, Qwen2Model):
        model.llm.model.forward = MethodType(partial(Qwen2Model_streamingllm_forward, model=model), model.llm.model)
    for i, decoder_layer in enumerate(model.llm.model.layers):
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2SdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2SdpaAttention_streamingllm_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")

def Qwen2SdpaAttention_streamingllm_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        model = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


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

@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2Model_streamingllm_forward(
    self,
    input_ids: torch.LongTensor = None,
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
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False


    init_num = model.init_num
    if inputs_embeds != None:
        local_window_num = int(inputs_embeds.shape[1] * model.length_rate) - init_num

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = SinkCache(window_length = init_num + local_window_num, num_sink_tokens= init_num )
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

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
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

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
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


"""
Merging forward misc
"""

### Merging ###
def replace_Qwen2_merging(model, sparsity=[0.1] * 28):
    model.sparsity = sparsity

    if isinstance(model.model, Qwen2Model):
        model.model.forward = MethodType(Qwen2Model_merging_forward, model.model)
        
    for i, decoder_layer in enumerate(model.model.layers):
        if isinstance(decoder_layer, Qwen2DecoderLayer):
            decoder_layer.forward = MethodType(Qwen2DecoderLayer_merging_forward, decoder_layer)
            
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2SdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2SdpaAttention_merging_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")

def Qwen2SdpaAttention_merging_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        model = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

    bsz, q_len, _ = hidden_states.size()

    ### start token merging
    def cosine_similarity(mat1, mat2):
        dot_product = torch.sum(mat1*mat2, dim=-1)
        norm_vec1 = torch.norm(mat1, dim=-1)
        norm_vec2 = torch.norm(mat2, dim=-1)
        return dot_product / (norm_vec1 * norm_vec2)

    device = hidden_states.device
    token_patch_type = model.patch_type.reshape(1, -1).to(device) 
    token_mask = None  # to store the merging positions in this layer 

    if q_len >1:
        # prefill
        sparsity = model.sparsity[self.layer_idx]
        frame_token_num = torch.sum(token_patch_type != TEXT_TOKEN).item()
        prune_num = math.floor(sparsity * frame_token_num)

        if prune_num > 0:
            # prefill token merging

            token_similarity = torch.full(
                (
                    bsz,
                    q_len,
                ),
                IGNORE_TOKEN,
                dtype=hidden_states.dtype,
                device=device,
            )

            assert bsz == 1, "Only support batch size 1"

            token_index_by_patch = []
            similarity_by_patch = []

            patch_num = model.patch_num # typically 14 * 15 = 210

            for i in range(patch_num):
                this_patch_token_index: torch.LongTensor = torch.where(
                    token_patch_type == i
                )[
                    1
                ]  # shape (q_len,)
                if this_patch_token_index.shape[-1] > 1:
                    this_patch_similarity = torch.cat(
                        (
                            torch.full(
                                size=(bsz, 1),
                                fill_value = IGNORE_TOKEN,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            ),
                            cosine_similarity(
                                hidden_states[:, this_patch_token_index[1:], :],
                                hidden_states[:, this_patch_token_index[:-1], :],
                            ),
                        ),
                        dim=-1,
                    )
                    similarity_by_patch.append(this_patch_similarity)
                    token_similarity[:, this_patch_token_index[1:]] = this_patch_similarity[
                        :, 1:
                    ]
                elif this_patch_token_index.shape[-1] == 1:
                    this_patch_similarity = torch.full(
                        size=(bsz, 1),
                        fill_value = IGNORE_TOKEN,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    
                    similarity_by_patch.append(this_patch_similarity)
                    token_similarity[:, this_patch_token_index] = torch.full(
                        size=(bsz, 1),
                        fill_value = IGNORE_TOKEN,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                else:
                    raise ValueError("No token in this patch")
                token_index_by_patch.append(this_patch_token_index.to(device))
            similarity_by_patch = torch.cat(similarity_by_patch, dim=-1)

            token_index_by_patch = torch.cat(token_index_by_patch, dim=0).reshape(
                1, -1
            )  # shape (batch_size, q_len),

            assert similarity_by_patch.shape[1] == token_index_by_patch.shape[1]

            # profile purpose
            if hasattr(model, "similarities"):
                model.similarities.append(token_similarity.detach().cpu())
            else:
                model.similarities = [token_similarity.detach().cpu()]

            topk_values, topk_indices = torch.topk(similarity_by_patch, prune_num)

            bsz_index = torch.arange(bsz, device=hidden_states.device)[:, None]
            mask_by_patch = torch.zeros(
                bsz,
                similarity_by_patch.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            mask_by_patch[bsz_index.to(hidden_states.device), topk_indices] = 1

            token_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            token_mask[bsz_index, token_index_by_patch[bsz_index, topk_indices]] = False

            last_merge_token_by_patch = find_contigious_latter_index(mask_by_patch)

            unique_merge_nums = [int(merge_num.item()) for merge_num in torch.unique(last_merge_token_by_patch.to(torch.long))]

            for merge_num in unique_merge_nums:
                if merge_num > 0:
                    batch_merge_indices, token_merge_indices = torch.where(
                        last_merge_token_by_patch == merge_num
                    )

                    token_merge_start_indices = token_merge_indices - merge_num  # 1D tensor

                    contigious_indices = (
                        token_merge_start_indices[:, None]
                        + torch.arange(
                            merge_num + 1,
                            dtype=torch.long,
                            device=device,
                        )[None, :]
                    )

                    hidden_states[
                        batch_merge_indices,
                        token_index_by_patch[
                            batch_merge_indices, token_merge_start_indices
                        ],
                    ] = hidden_states[
                        batch_merge_indices[:, None],
                        token_index_by_patch[
                            batch_merge_indices[:, None], contigious_indices
                        ],
                    ].mean(
                        dim=1
                    )
            # here only bsz=1

            # update patch type
            model.patch_type = model.patch_type.to(device)[token_mask].reshape(bsz, -1)
            hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, self.hidden_size)
    ### end token merging

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    if q_len > 1:
        q_len = hidden_states.shape[1]

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        if q_len>1 and prune_num > 0:
            ### also prune position_embeddings according to mask
            position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
            position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
            ###end pruning position_embeddings
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value, token_mask

def Qwen2DecoderLayer_merging_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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

        hidden_states = self.input_layernorm(hidden_states).to(residual.device)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, mask = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        ### implement mask on residual to match the shape of hidden_states
        if mask is not None:
            device=residual.device
            mask=mask.to(device)
            residual = residual[:, mask[0], :]
        ### end masking residual

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        residual.to(hidden_states.device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2Model_merging_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

### Merge then prune ###

def replace_Qwen2_merge_then_fastv(model, sparsity = [0.1]*28, fastv_k = 3, fastv_r = 0.5):
    model.sparsity = sparsity
    model.fastv_k = fastv_k
    model.fastv_r = fastv_r 
    
    if isinstance(model.model, Qwen2Model):
        model.model.forward = MethodType(partial(Qwen2Model_merge_then_fastv_forward, model=model), model.model)
    for i, decoder_layer in enumerate(model.model.layers):
        if isinstance(decoder_layer, Qwen2DecoderLayer):
            decoder_layer.forward=MethodType(Qwen2DecoderLayer_merge_then_fastv_forward, decoder_layer)
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2SdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2SdpaAttention_merge_then_fastv_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")

@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2Model_merge_then_fastv_forward(
    self,
    input_ids: torch.LongTensor = None,
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
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

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
    # fastv_constants
    device = self.device
    #seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1] (here because cache position in minicpmv is not none,so past_seen_tokens is not defined )
    FASTV_k = model.fastv_k # the layer_idx to prune
    FASTV_r = model.fastv_r # the pruning ratio
    FASTV_image_token_start_index = model.image_token_start_index.item()
    FASTV_image_token_length = model.image_token_length.item()

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # pruning hidden states after layer k, no kv cache
        if use_cache:
            if hidden_states.shape[1] != 1:
                if layer_idx<FASTV_k:
                    pruned_attention_mask = causal_mask

                elif layer_idx==FASTV_k:
                    # update FASTV_image_token_length
                    FASTV_image_token_length = (model.image_token_length - (model.original_length - hidden_states.shape[1])).item()
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
                    keep_indexs = torch.cat( (torch.arange(FASTV_image_token_start_index,device=device), top_attention_rank_index, torch.arange(FASTV_image_token_start_index+FASTV_image_token_length, hidden_states.shape[1],device=device)))
                    # sort index
                    keep_indexs = keep_indexs.sort().values
                    # update seq length
                    new_seq_length = keep_indexs.shape[0]
                    # filter hidden states
                    hidden_states = hidden_states[:,keep_indexs,:] 
                    # update position ids
                    position_ids = keep_indexs.unsqueeze(0)
                    # update position embeddings
                    position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                    position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
                    # update attention mask
                    pruned_attention_mask =None

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
    ###finish inplementing fastv
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
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def Qwen2SdpaAttention_merge_then_fastv_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        model = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    sparsity = model.sparsity[self.layer_idx]
    if sparsity > 0:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                    "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                    'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

        bsz, q_len, _ = hidden_states.size()
    ### start token merging
        def cosine_similarity(mat1, mat2):
            dot_product = torch.sum(mat1*mat2, dim=-1)
            norm_vec1 = torch.norm(mat1, dim=-1)
            norm_vec2 = torch.norm(mat2, dim=-1)
            return dot_product / (norm_vec1 * norm_vec2)

        device = hidden_states.device
        token_patch_type = model.patch_type.reshape(1, -1).to(device)
        token_mask = None

        if q_len >1:
            # prefill
            sparsity = model.sparsity[self.layer_idx]
            frame_token_num = torch.sum(token_patch_type != TEXT_TOKEN).item()
            prune_num = math.floor(sparsity * frame_token_num)

            if prune_num > 0:
                # prefill token merging

                token_similarity = torch.full(
                    (
                        bsz,
                        q_len,
                    ),
                    IGNORE_TOKEN,
                    dtype=hidden_states.dtype,
                    device=device,
                )

                token_merge_scale = model.token_merge_scale

                assert bsz == 1, "Only support batch size 1"

                token_index_by_patch = []
                similarity_by_patch = []

                patch_num=model.patch_num # typically 14 * 15 = 210

                for i in range(patch_num):
                    this_patch_token_index: torch.LongTensor = torch.where(
                        token_patch_type == i
                    )[
                        1
                    ]  # shape (q_len,)
                    if this_patch_token_index.shape[-1] > 1:
                        this_patch_similarity = torch.cat(
                            (
                                torch.full(
                                    size=(bsz, 1),
                                    fill_value = IGNORE_TOKEN,
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device,
                                ),
                                cosine_similarity(
                                    hidden_states[:, this_patch_token_index[1:], :],
                                    hidden_states[:, this_patch_token_index[:-1], :],
                                ),
                            ),
                            dim=-1,
                        )
                        similarity_by_patch.append(this_patch_similarity)
                        token_similarity[:, this_patch_token_index[1:]] = this_patch_similarity[
                            :, 1:
                        ]
                    elif this_patch_token_index.shape[-1] == 1:
                        this_patch_similarity = torch.full(
                            size=(bsz, 1),
                            fill_value = IGNORE_TOKEN,
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                        
                        similarity_by_patch.append(this_patch_similarity)
                        token_similarity[:, this_patch_token_index] = torch.full(
                            size=(bsz, 1),
                            fill_value = IGNORE_TOKEN,
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                    else:
                        raise ValueError("No token in this patch")
                    token_index_by_patch.append(this_patch_token_index.to(device))
                similarity_by_patch = torch.cat(similarity_by_patch, dim=-1)

                token_index_by_patch = torch.cat(token_index_by_patch, dim=0).reshape(
                    1, -1
                )  # shape (batch_size, q_len),

                assert similarity_by_patch.shape[1] == token_index_by_patch.shape[1]

                # profile purpose
                if hasattr(model, "similarities"):
                    model.similarities.append(token_similarity.detach().cpu())
                else:
                    model.similarities = [token_similarity.detach().cpu()]

                topk_values, topk_indices = torch.topk(similarity_by_patch, prune_num)

                bsz_index = torch.arange(bsz, device=hidden_states.device)[:, None]
                mask_by_patch = torch.zeros(
                    bsz,
                    similarity_by_patch.shape[1],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                mask_by_patch[bsz_index.to(hidden_states.device), topk_indices] = 1

                token_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
                token_mask[bsz_index, token_index_by_patch[bsz_index, topk_indices]] = False

                last_merge_token_by_patch = find_contigious_latter_index(mask_by_patch)

                unique_merge_nums = [int(merge_num.item()) for merge_num in torch.unique(last_merge_token_by_patch.to(torch.long))]

                for merge_num in unique_merge_nums:
                    if merge_num > 0:
                        batch_merge_indices, token_merge_indices = torch.where(
                            last_merge_token_by_patch == merge_num
                        )

                        token_merge_start_indices = token_merge_indices - merge_num  # 1D tensor

                        contigious_indices = (
                            token_merge_start_indices[:, None]
                            + torch.arange(
                                merge_num + 1,
                                dtype=torch.long,
                                device=device,
                            )[None, :]
                        )

                        hidden_states[
                            batch_merge_indices,
                            token_index_by_patch[
                                batch_merge_indices, token_merge_start_indices
                            ],
                        ] = hidden_states[
                            batch_merge_indices[:, None],
                            token_index_by_patch[
                                batch_merge_indices[:, None], contigious_indices
                            ],
                        ].mean(
                            dim=1
                        )

                        token_merge_scale[
                            batch_merge_indices,
                            token_index_by_patch[
                                batch_merge_indices, token_merge_start_indices
                            ],
                        ] = token_merge_scale[
                            batch_merge_indices[:, None],
                            token_index_by_patch[
                                batch_merge_indices[:, None], contigious_indices
                            ],
                        ].sum(
                            dim=1
                        )

                # here only bsz=1
                # update patch type
                model.patch_type = model.patch_type.to(device)[token_mask].reshape(bsz, -1)
                model.token_merge_scale = token_merge_scale[token_mask].reshape(bsz, -1)

                hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, self.hidden_size)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if q_len > 1:
            q_len = hidden_states.shape[1]

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            if q_len>1 and prune_num > 0:
                position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
                position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        ### storing attnetion weights
        else:

            def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

                if is_causal:
                    assert attn_mask is None
                    temp_mask = torch.ones(L, S, dtype=torch.bool,device=query.device).tril(diagonal=0)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query.dtype)

                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias += attn_mask

                if enable_gqa:
                    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

                attn_weight = query@ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight=attn_weight
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

                return (attn_weight @ value).to(query.dtype), attn_weight
            
            attn_output, attn_weights = scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        ### finish storing attention weights

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, token_mask

    else:

    ### end token merging
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.to(query_states.device), sin.to(query_states.device))

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
        ### storing attnetion weights
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

            def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

                if is_causal:
                    assert attn_mask is None
                    temp_mask = torch.ones(L, S, dtype=torch.bool,device=query.device).tril(diagonal=0)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query.dtype)

                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias += attn_mask

                if enable_gqa:
                    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

                attn_weight = query@ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight=attn_weight
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

                return (attn_weight @ value).to(query.dtype), attn_weight
            
            attn_output, attn_weights = scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        ### finish storing attention weights


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, None

def Qwen2DecoderLayer_merge_then_fastv_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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

        hidden_states = self.input_layernorm(hidden_states).to(residual.device)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, mask = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        ### implement mask on residual to match the shape of hidden_states
        if mask is not None:
            device=residual.device
            mask=mask.to(device)
            residual = residual[:, mask[0], :]
        ### end masking residual
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        residual.to(hidden_states.device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if self_attn_weights!=None:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

### Prune then merge ###

def replace_Qwen2_fastv_then_merge(model, fastv_k = 2, fastv_r = 0.75, merging_sparsity = 0.3):
    model.fastv_k = fastv_k
    model.fastv_r = fastv_r
    model.merging_sparsity = merging_sparsity

    if isinstance(model.model, Qwen2Model):
        model.model.forward = MethodType(partial(Qwen2Model_fastv_then_merge_forward, model=model), model.model)
    for i, decoder_layer in enumerate(model.model.layers):
        if isinstance(decoder_layer, Qwen2DecoderLayer):
            decoder_layer.forward=MethodType(Qwen2DecoderLayer_fastv_then_merge_forward, decoder_layer)
        qwen2_attention_instance = decoder_layer.self_attn
        if isinstance(qwen2_attention_instance, Qwen2SdpaAttention):
            qwen2_attention_instance.forward = MethodType(partial(Qwen2SdpaAttention_fastv_then_merge_forward, model=model), qwen2_attention_instance)
        else:
            raise TypeError("language model is not Qwen2.")

@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2Model_fastv_then_merge_forward(
    self,
    input_ids: torch.LongTensor = None,
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
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    position_embeddings = list(position_embeddings)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None


    ### implement fastv
 
    device = self.device
    FASTV_image_token_start_index = model.image_token_start_index.item()
    FASTV_image_token_length = model.image_token_length.item()
    FASTV_k = model.fastv_k
    FASTV_r = model.fastv_r
    #seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1]    (here because cache position in minicpmv is not none,so past_seen_tokens is not defined )

    model.sparsity_list = []
    
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  

        if layer_idx == FASTV_k and hidden_states.shape[1] > 1: 
            # update FASTV_image_token_length
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
            # update patch type
            model.patch_type = model.patch_type[:,keep_indexs]
            # update position ids
            position_ids = keep_indexs.unsqueeze(0)
            # update position embeddings
            position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
            position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
    
            position_embeddings = list(position_embeddings)
            cache_position = cache_position[:new_seq_length]


        layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

        if layer_idx == FASTV_k - 1 and hidden_states.shape[1] > 1:
            output_attentions = True
        else:
            output_attentions = False
    ###finish inplementing fastv

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
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def Qwen2SdpaAttention_fastv_then_merge_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        model = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
    
    bsz, q_len, _ = hidden_states.size()
    # start token merging
    
    def cosine_similarity(mat1, mat2):
        dot_product = torch.sum(mat1*mat2, dim=-1)
        norm_vec1 = torch.norm(mat1, dim=-1)
        norm_vec2 = torch.norm(mat2, dim=-1)
        return dot_product / (norm_vec1 * norm_vec2)

    device = hidden_states.device
    token_patch_type = model.patch_type.reshape(1, -1).to(device)
    token_mask = None

    if q_len >1 and self.layer_idx == model.fastv_k + 1:
        # prefill
        sparsity = model.merging_sparsity
        frame_token_num = torch.sum(token_patch_type != TEXT_TOKEN).item()
        prune_num = math.floor(sparsity * frame_token_num)

        if prune_num > 0:
            # prefill token merging

            token_similarity = torch.full(
                (
                    bsz,
                    q_len,
                ),
                IGNORE_TOKEN,
                dtype=hidden_states.dtype,
                device=device,
            )


            assert bsz == 1, "Only support batch size 1"

            token_index_by_patch = []
            similarity_by_patch = []

            patch_num=model.patch_num # typically 14 * 15 = 210

            for i in range(patch_num):
                this_patch_token_index: torch.LongTensor = torch.where(
                    token_patch_type == i
                )[
                    1
                ]  # shape (q_len,)
                if this_patch_token_index.shape[-1] > 1:
                    this_patch_similarity = torch.cat(
                        (
                            torch.full(
                                size=(bsz, 1),
                                fill_value = IGNORE_TOKEN,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            ),
                            cosine_similarity(
                                hidden_states[:, this_patch_token_index[1:], :],
                                hidden_states[:, this_patch_token_index[:-1], :],
                            ),
                        ),
                        dim=-1,
                    )
                    similarity_by_patch.append(this_patch_similarity)
                    token_similarity[:, this_patch_token_index[1:]] = this_patch_similarity[
                        :, 1:
                    ]
                elif this_patch_token_index.shape[-1] == 1:
                    this_patch_similarity = torch.full(
                        size=(bsz, 1),
                        fill_value = IGNORE_TOKEN,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    
                    similarity_by_patch.append(this_patch_similarity)
                    token_similarity[:, this_patch_token_index] = torch.full(
                        size=(bsz, 1),
                        fill_value = IGNORE_TOKEN,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                else:
                    raise ValueError("No token in this patch")
                token_index_by_patch.append(this_patch_token_index.to(device))
            similarity_by_patch = torch.cat(similarity_by_patch, dim=-1)

            token_index_by_patch = torch.cat(token_index_by_patch, dim=0).reshape(
                1, -1
            )  # shape (batch_size, q_len),

            assert similarity_by_patch.shape[1] == token_index_by_patch.shape[1]

            # profile purpose
            if hasattr(model, "similarities"):
                model.similarities.append(token_similarity.detach().cpu())
            else:
                model.similarities = [token_similarity.detach().cpu()]

            topk_values, topk_indices = torch.topk(similarity_by_patch, prune_num)

            bsz_index = torch.arange(bsz, device=hidden_states.device)[:, None]
            mask_by_patch = torch.zeros(
                bsz,
                similarity_by_patch.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            mask_by_patch[bsz_index.to(hidden_states.device), topk_indices] = 1

            token_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            token_mask[bsz_index, token_index_by_patch[bsz_index, topk_indices]] = False

            last_merge_token_by_patch = find_contigious_latter_index(mask_by_patch)

            unique_merge_nums = [int(merge_num.item()) for merge_num in torch.unique(last_merge_token_by_patch.to(torch.long))]

            for merge_num in unique_merge_nums:
                if merge_num > 0:
                    batch_merge_indices, token_merge_indices = torch.where(
                        last_merge_token_by_patch == merge_num
                    )

                    token_merge_start_indices = token_merge_indices - merge_num  # 1D tensor

                    contigious_indices = (
                        token_merge_start_indices[:, None]
                        + torch.arange(
                            merge_num + 1,
                            dtype=torch.long,
                            device=device,
                        )[None, :]
                    )

                    hidden_states[
                        batch_merge_indices,
                        token_index_by_patch[
                            batch_merge_indices, token_merge_start_indices
                        ],
                    ] = hidden_states[
                        batch_merge_indices[:, None],
                        token_index_by_patch[
                            batch_merge_indices[:, None], contigious_indices
                        ],
                    ].mean(
                        dim=1
                    )

            # here only bsz=1
            # update patch type
            model.patch_type = model.patch_type.to(device)[token_mask].reshape(bsz, -1)
            hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, self.hidden_size)
    ### end token merging
          
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    if q_len > 1:
        q_len = hidden_states.shape[1]

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        if q_len>1 and token_mask != None:
            ### also prune position_embeddings according to mask
            position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
            position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
            ###end pruning position_embeddings
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
    
    ### start storing attn_weights
    attn_weights = None
    if q_len > 1 and self.layer_idx == model.fastv_k - 1:
        def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

                if is_causal:
                    assert attn_mask is None
                    temp_mask = torch.ones(L, S, dtype=torch.bool,device=query.device).tril(diagonal=0)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query.dtype)

                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias += attn_mask

                if enable_gqa:
                    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)


                attn_weight = query@ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight=attn_weight
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

                return (attn_weight @ value).to(query.dtype), attn_weight
            
        attn_output, attn_weights = scaled_dot_product_attention(
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
    ### end storing attn_weights

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value, token_mask

def Qwen2DecoderLayer_fastv_then_merge_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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

        hidden_states = self.input_layernorm(hidden_states).to(residual.device)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, mask = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        
        ### implement mask on residual to match the shape of hidden_states
        if mask is not None:
            device=residual.device
            mask=mask.to(device)
            residual = residual[:, mask[0], :]
        ### end masking residual
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        residual.to(hidden_states.device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        ### adding attn_weights if needed
        if self_attn_weights != None:
            outputs += (self_attn_weights,)
        ### end adding

        if use_cache:
            outputs += (present_key_value,)

        return outputs
