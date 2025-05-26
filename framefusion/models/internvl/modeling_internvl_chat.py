import torch 
from typing import Optional, Callable
from transformers import GenerationConfig
TEXT_TOKEN = -1

@torch.no_grad()
def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
    
    ### framefusion begins
    def count_consecutive_false(selected):
        false_indices = (selected == False).nonzero().squeeze()
        if false_indices.numel() == 0:
            return torch.tensor([], dtype=torch.long)
        
        diffs = torch.diff(false_indices)
        break_points = (diffs != 1).nonzero().squeeze()
        if break_points.dim() == 0:
            break_points = break_points.unsqueeze(0)
        segments = []
        start = 0
        for bp in break_points:
            segments.append(false_indices[start:bp+1])
            start = bp + 1
        segments.append(false_indices[start:])
        lengths = torch.tensor([len(seg) for seg in segments])
        
        return lengths

    if input_ids.shape[-1] != 1:
        image_token_start_index = torch.where(selected)[0][0]
        image_token_end_index = torch.where(selected)[0][-1]
        image_token_length = image_token_end_index - image_token_start_index + 1
        patch_num = vit_embeds.shape[1]
        patch_num = int(patch_num)
        original_length = N
        n_frames = vit_embeds.shape[0]

        text_length_list = count_consecutive_false(selected)[1:-1]

        patch_type = [TEXT_TOKEN] * image_token_start_index 
        for i in range(n_frames-1):
            patch_type = patch_type + list(range(patch_num)) + [TEXT_TOKEN] * text_length_list[i]
        patch_type = patch_type + list(range(patch_num)) + [TEXT_TOKEN] * (original_length - image_token_end_index - 1)

        patch_type = torch.tensor([patch_type], device=input_embeds.device)

        self.patch_num = patch_num
        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.original_length = original_length

        if not hasattr(self, 'mode'):
            self.framefusion.prepare(patch_type, patch_num, image_token_start_index, image_token_end_index, image_token_length, original_length)
    ### framefusion ends

    

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs