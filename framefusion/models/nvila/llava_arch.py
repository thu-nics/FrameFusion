import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from llava.constants import IGNORE_INDEX
#from vila_llava.model.llava_arch import __embed_media_tokens
TEXT_TOKEN = -1

def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embedding function for modified from llava/model/llava_arch.py, `LlavaMetaForCausalLM._embed`
        """
        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        media_embeds = self._LlavaMetaForCausalLM__embed_media_tokens(media, media_config)

        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0

        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        ### start modification by framefusion ###
        if isinstance(self.config.video_encoder, dict) and 'pool_sizes' in self.config.video_encoder:
            pool_sizes = self.config.video_encoder['pool_sizes'][0][0]
        else:
            pool_sizes = 1

        if 'video' in media_embeds:
            num_frames = media['video'][0].shape[0]
            num_frames = num_frames / pool_sizes
            length = text_embeds[0].shape[0] + media_embeds['video'][0].shape[0] - 1
            patch_type = torch.full((batch_size, length), TEXT_TOKEN, dtype=torch.long, device = text_embeds[0].device)
            patch_num = media_embeds['video'][0].shape[0] / num_frames
        if 'image' in media_embeds:
            length = text_embeds[0].shape[0] + media_embeds['image'][0].shape[0] - 1
            patch_type = torch.full((batch_size, length), TEXT_TOKEN, dtype=torch.long, device = text_embeds[0].device)
            patch_num = 1
            num_frames = media_embeds['image'][0].shape[0]
        ### end modification by framefusion ###


        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    end = pos + 1
                    name = media_tokens[input_ids[k][pos].item()]
                    input = media_embeds[name].popleft()
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
                else:
                    end = pos
                    while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]
                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))

            ### start modification by framefusion ###
            seq = torch.arange(patch_num, device = patch_type.device).repeat(int(num_frames))
            patch_type[k, inputs_mk[0].shape[0]:inputs_mk[0].shape[0]+inputs_mk[1].shape[0]] = seq
            ### end modification by framefusion ###

        ### Framefusion edit start ###
        image_token_start_index = torch.argmax((patch_type >=0).int(), dim=1)
        image_token_end_index = patch_type.shape[1]-1-torch.argmax((torch.flip(patch_type, dims=[1]) >=0).int(), dim=1)
        original_length = patch_type.shape[1]
        image_token_length = image_token_end_index - image_token_start_index + 1

        self.patch_type = patch_type
        self.patch_num = patch_num
        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.original_length = original_length
        self.framefusion.prepare(patch_type, patch_num, image_token_start_index, image_token_end_index, image_token_length, original_length)
        ### Framefusion edit end ###

        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed
        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed!")

        # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self._LlavaMetaForCausalLM__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return self._LlavaMetaForCausalLM__batchify_sequence(inputs, labels)

