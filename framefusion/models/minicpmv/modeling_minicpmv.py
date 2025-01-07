import torch
import math

TEXT_TOKEN = -1


def get_vllm_embedding(self, data):
    if "vision_hidden_states" not in data:
        dtype = self.llm.model.embed_tokens.weight.dtype
        device = self.llm.model.embed_tokens.weight.device
        tgt_sizes = data["tgt_sizes"]
        pixel_values_list = data["pixel_values"]
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        # exist image
        if all_pixel_values:
            tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
            for i in range(B):
                patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

            vision_batch_size = self.config.vision_batch_size
            all_pixel_values = all_pixel_values.type(dtype)
            if B > vision_batch_size:
                hs = []
                for i in range(0, B, vision_batch_size):
                    start_idx = i
                    end_idx = i + vision_batch_size
                    tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                    hs.append(tmp_hs)
                vision_embedding = torch.cat(hs, dim=0)
            else:
                vision_embedding = self.vpm(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes).last_hidden_state
            vision_embedding = self.resampler(vision_embedding, tgt_sizes)

            start = 0
            for pixel_values in pixel_values_list:
                img_cnt = len(pixel_values)
                if img_cnt > 0:
                    vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                    start += img_cnt
                else:
                    vision_hidden_states.append([])
        else:  # no image
            if self.training:
                dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
            else:
                dummy_feature = []
            for _ in range(len(pixel_values_list)):
                vision_hidden_states.append(dummy_feature)

    else:
        vision_hidden_states = data["vision_hidden_states"]

    if hasattr(self.llm.config, "scale_emb"):
        vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
    else:
        vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

    vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states]

    bs = len(data["input_ids"])
    for i in range(bs):
        cur_vs_hs = vision_hidden_states[i]
        if len(cur_vs_hs) > 0:
            cur_vllm_emb = vllm_embedding[i]
            cur_image_bound = data["image_bound"][i]
            if len(cur_image_bound) > 0:
                image_indices = torch.stack([torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]).to(vllm_embedding.device)

                cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]), cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
            elif self.training:
                cur_vllm_emb += cur_vs_hs[0].mean() * 0

    ### FRAMEFUSION START ###
    assert bs == 1
    patch_type = torch.full((bs, vllm_embedding.shape[1]), TEXT_TOKEN, dtype=torch.long, device=vllm_embedding.device)
    num_frames = self.num_frames

    image_bound = data["image_bound"][0]
    patch_per_frame = image_bound.shape[0] // num_frames
    token_per_frame = image_bound[patch_per_frame, 0] - image_bound[0, 0]
    patch_type[i, image_bound[0, 0] : (image_bound[-1, 1] + 2)] = torch.arange(0, image_bound[-1, 1] - image_bound[0, 0] + 2, device=patch_type.device) % token_per_frame

    patch_num = token_per_frame
    image_token_start_index = torch.argmax((patch_type >= 0).int(), dim=1)
    image_token_end_index = patch_type.shape[1] - 1 - torch.argmax((torch.flip(patch_type, dims=[1]) >= 0).int(), dim=1)
    original_length = patch_type.shape[1]
    image_token_length = image_token_end_index - image_token_start_index + 1

    self.framefusion.prepare(patch_type, patch_num, image_token_start_index, image_token_end_index, image_token_length, original_length)
    ### FRAMEFUSION END ###

    return vllm_embedding, vision_hidden_states
