from typing import List
import torch
from torch import nn
from framefusion.main import cosine_similarity, find_contigious_latter_index

TEXT_TOKEN = -1
IGNORE_TOKEN = -2

class FrameFusionAdjacent(nn.Module):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1):
        super(FrameFusionAdjacent, self).__init__()
        self.cost = cost
        self.similarity_lower_bound = similarity_lower_bound
        self.ratio_lower_bound = ratio_lower_bound

    def prepare(self, patch_type, patch_num, image_token_start_index, image_token_end_index, image_token_length, original_length, finish_merging = False, finish_pruning = False, sparsity_list: List = None):
        self.patch_type = patch_type
        self.patch_num = patch_num
        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.original_length = original_length
        self.finish_merging = finish_merging
        self.finish_pruning = finish_pruning
        if sparsity_list is None:
            self.sparsity_list = []
        else:
            self.sparsity_list = sparsity_list

    def forward(self, hidden_states, position_embeddings, attention_mask, self_attn_weights = None):
        """
        This is the forward method of the FrameFusion_Adjacent class.

        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).
            self_attn_weights (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).

        Returns:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).
        """
        bsz, q_len, hidden_size = hidden_states.size()
        device = hidden_states.device    

        # pruning
        if q_len >1 and self.finish_merging == True and self.finish_pruning == False:
            image_token_pruning_start_index = self.image_token_start_index.item()
            image_token_pruning_length = self.image_token_length
            # update image_token_pruning_length
            image_token_pruning_length = (self.image_token_length - (self.original_length - q_len))

            last_layer_attention = self_attn_weights
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=(1,2))[0]
            last_layer_attention_avg_image = last_layer_attention_avg[image_token_pruning_start_index:image_token_pruning_start_index+image_token_pruning_length]
            
            pruning_ratio = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            top_attention_rank_index = last_layer_attention_avg_image.topk(round(image_token_pruning_length*(1-pruning_ratio))).indices + image_token_pruning_start_index
            
            keep_indexs = torch.cat( (torch.arange(image_token_pruning_start_index,device=device), top_attention_rank_index, torch.arange(image_token_pruning_start_index+image_token_pruning_length, q_len, device=device)))
            keep_indexs = keep_indexs.sort().values
            
            hidden_states = hidden_states[:,keep_indexs,:] 
            position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
            position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
            if attention_mask != None:
                attention_mask = attention_mask[:,:,keep_indexs,:][:,:,:,keep_indexs]
            self.finish_pruning = True

        # merging
        if q_len >1 and (not self.finish_merging):
            # align devices
            self.patch_type = self.patch_type.to(device)

            # prefill
            sparsity_upper_bound = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            similarity_by_patch, token_index_by_patch = self.compute_similarity_and_token_index_by_patch(hidden_states, self.patch_type) # only support bsz = 1
            
            frame_token_num = torch.sum(self.patch_type != TEXT_TOKEN).item()
            merge_index_by_patch = torch.where(similarity_by_patch >= self.similarity_lower_bound)[1]
            above_k_ratio = merge_index_by_patch.shape[0] / frame_token_num

            if above_k_ratio < sparsity_upper_bound:
                self.sparsity_list.append(above_k_ratio)

                if above_k_ratio < self.ratio_lower_bound:
                    self.finish_merging = True
            else:
                topk_values, topk_indices = torch.topk(similarity_by_patch, int(sparsity_upper_bound*frame_token_num))
                topk_indices, _ = torch.sort(topk_indices)
                merge_index_by_patch = topk_indices[0]

                self.finish_merging = True
                self.finish_pruning = True
                
            hidden_states, token_mask = self.merge_tokens_and_get_mask(hidden_states, similarity_by_patch, token_index_by_patch, merge_index_by_patch)
            # here only bsz=1
            # update patch type
            self.patch_type = self.patch_type.to(device)[token_mask].reshape(bsz, -1)
            hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, hidden_size)
            position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
            position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
            if attention_mask is not None:
                attention_mask = attention_mask[:,:,token_mask[0],:][:,:,:,token_mask[0]]

        return hidden_states, position_embeddings, attention_mask

    @staticmethod
    def compute_similarity_and_token_index_by_patch(hidden_states, token_patch_type):
        """
        Compute the similarity between consecutive non-text tokens, regardless of their patch type.

        Args:
            hidden_states (torch.Tensor): 
                A tensor of shape (batch_size, sequence_length, hidden_size).
            token_patch_type (torch.Tensor): 
                A tensor indicating the patch type of each token in the sequence. Text tokens are set to TEXT_TOKEN.

        Returns:
            similarity_by_patch (torch.Tensor): 
                A tensor of shape (batch_size, non_text_sequence_length) containing the cosine similarity between consecutive non-text tokens.
            token_index_by_patch (torch.Tensor): 
                A tensor of shape (batch_size, non_text_sequence_length) containing the original token indices of non-text tokens.
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        assert bsz == 1, "Only support batch size 1"

        # Get indices of non-text tokens
        non_text_indices = torch.where(token_patch_type != TEXT_TOKEN)[1]
        token_index_by_patch = non_text_indices[None, :]  # Add batch dimension

        # Compute similarity between consecutive non-text tokens
        similarity_by_patch = cosine_similarity(
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, :-1], :
            ],
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, 1:], :
            ],
        )

        # Add IGNORE_TOKEN at the start to match dimensions
        similarity_by_patch = torch.cat(
            (
                torch.full(
                    size=(bsz, 1),
                    fill_value=IGNORE_TOKEN,
                    dtype=hidden_states.dtype,
                    device=device,
                ),
                similarity_by_patch,
            ),
            dim=1,
        )

        assert similarity_by_patch.shape[1] == token_index_by_patch.shape[1]
        return similarity_by_patch, token_index_by_patch

    @staticmethod
    def merge_tokens_and_get_mask(hidden_states: torch.Tensor, similarity_by_patch, token_index_by_patch, merge_index_by_patch):
        """
        Merge tokens and get a mask indicating which tokens to keep.

        Args:
            hidden_states (torch.Tensor): 
                A tensor of shape (batch_size, sequence_length, hidden_size)
            similarity_by_patch (torch.Tensor): 
                A tensor of shape (batch_size, sequence_length) containing the cosine similarity between consecutive non-text tokens.
            token_index_by_patch (torch.Tensor): 
                A tensor of shape (batch_size, sequence_length) containing the original token indices of non-text tokens.
            merge_index_by_patch (torch.Tensor): 
                A tensor containing the indices of tokens to be merged.

        Returns:
            hidden_states (torch.Tensor): A tensor containing the hidden states of the tokens after merging.
            keep_mask (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length) indicating which tokens in the original sequence should be kept after merging.
        """
        device = hidden_states.device
        if merge_index_by_patch.shape[0] == 0:
            keep_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            return hidden_states, keep_mask
        
        bsz, q_len, _ = hidden_states.size()
        bsz_index = torch.arange(bsz, device=hidden_states.device)[:, None]
        merge_mask_by_patch: torch.LongTensor = torch.zeros(
            bsz,
            similarity_by_patch.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        merge_mask_by_patch[bsz_index, merge_index_by_patch] = 1
        last_merge_token_by_patch = find_contigious_latter_index(merge_mask_by_patch)

        keep_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
        keep_mask[bsz_index, token_index_by_patch[bsz_index, merge_index_by_patch]] = False

        # noqa: batch size = 1
        unique_merge_nums = torch.sort(torch.unique(last_merge_token_by_patch.to(torch.long))).values
        unique_merge_nums = (unique_merge_nums[1:] if (unique_merge_nums[0] == 0).item() else unique_merge_nums)

        merge_num_indices, token_merge_index_in_patch = torch.where(
            last_merge_token_by_patch == unique_merge_nums[:, None]
        )

        merge_nums = unique_merge_nums[merge_num_indices]
        token_merge_start_index_in_patch = token_merge_index_in_patch - merge_nums
        token_merge_member_start_index_in_patch = torch.repeat_interleave(token_merge_start_index_in_patch, merge_nums)

        merge_member_length = torch.sum(merge_nums)
        merge_member_contigious_sequence = torch.arange(1, merge_member_length + 1, device = device)

        merge_nums_cumulative_counts = torch.cumsum(merge_nums, dim=0)
        merge_nums_start = torch.cat((torch.tensor([0], device = device), merge_nums_cumulative_counts[:-1]))

        contigious_sequence_by_merge_nums = merge_member_contigious_sequence - torch.repeat_interleave(merge_nums_start, merge_nums)

        token_merge_member_index_in_patch = token_merge_member_start_index_in_patch + contigious_sequence_by_merge_nums

        # noqa: this function may have numerical instability
        hidden_states.index_add_(
            dim = 1,
            index = token_index_by_patch[0, token_merge_member_start_index_in_patch],
            source = hidden_states[
                bsz_index,
                token_index_by_patch[bsz_index, token_merge_member_index_in_patch],
            ]
        )  

        # divide to get average
        hidden_states[
            bsz_index,
            token_index_by_patch[bsz_index, token_merge_start_index_in_patch],
        ] /= (merge_nums[None, :, None] + 1)

        return hidden_states, keep_mask

    @staticmethod
    def _compute_pruning_ratio(sparsity_list, cost, num_layers = 28):
        """
        Args:
            sparsity_list (list): A list containing the sparsity values of the model's first few layers.
            cost (float): The total computation budget given by the user.
            num_layers (int, optional): The number of layers in the model. 

        Returns:
            float: the required sparsity for the next layer to achieve the given cost
        """
        list_length = len(sparsity_list)
        s = 1
        total_calcution =0
        for i in range(list_length):
            s *= (1 - sparsity_list[i])
            total_calcution += s
        remain_calcution = num_layers * cost - total_calcution
        if remain_calcution < 0:
            raise ValueError("The cost is too small")
        if remain_calcution/((num_layers-list_length)*s) > 1:
            return 0
        return 1 - (remain_calcution/((num_layers-list_length)*s))