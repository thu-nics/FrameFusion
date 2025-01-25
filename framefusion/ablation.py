from typing import List
import torch
from torch import nn
from framefusion.main import cosine_similarity, find_contigious_latter_index, FrameFusion

TEXT_TOKEN = -1
IGNORE_TOKEN = -2

class FrameFusionAdjacent(FrameFusion):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1):
        super(FrameFusionAdjacent, self).__init__(cost, similarity_lower_bound, ratio_lower_bound)

    @staticmethod
    def compute_similarity_and_token_index_by_patch(hidden_states, token_patch_type, patch_num):
        """
        Compute the similarity between consecutive non-text tokens, regardless of their patch type.

        Args:
            hidden_states (torch.Tensor): 
                A tensor of shape (batch_size, sequence_length, hidden_size).
            token_patch_type (torch.Tensor): 
                A tensor indicating the patch type of each token in the sequence. Text tokens are set to TEXT_TOKEN.
            patch_num (int): 
                The number of patches in the sequence. Not used in this method.

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

class FrameFusionRandom(FrameFusion):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1):
        super(FrameFusionRandom, self).__init__(cost, similarity_lower_bound, ratio_lower_bound)

    @staticmethod
    def compute_similarity_and_token_index_by_patch(hidden_states, token_patch_type, patch_num):
        """
        Compute similarity for N randomly selected token pairs, where N equals the number of non-text tokens.
        For each pair, mark the similarity of the former token as IGNORE_TOKEN and assign the similarity to the latter token.

        Args:
            hidden_states (torch.Tensor): 
                A tensor of shape (batch_size, sequence_length, hidden_size).
            token_patch_type (torch.Tensor): 
                A tensor indicating the patch type of each token in the sequence. Text tokens are set to TEXT_TOKEN.
            patch_num (int): 
                The number of patches in the sequence. Not used in this method.

        Returns:
            similarity_by_patch (torch.Tensor): 
                A tensor of shape (batch_size, 2*non_text_sequence_length) containing the cosine similarity between randomly paired tokens.
            token_index_by_patch (torch.Tensor): 
                A tensor of shape (batch_size, 2*non_text_sequence_length) containing the original token indices of non-text tokens.
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        assert bsz == 1, "Only support batch size 1"

        # Get indices of non-text tokens
        non_text_indices = torch.where(token_patch_type != TEXT_TOKEN)[1]
        num_sample_pairs = len(non_text_indices)

        if num_sample_pairs >= 2:  # Need at least 2 tokens to form pairs
            # Sample N indices with replacement for N pairs
            sampled_indices = torch.randint(
                low=0,
                high=num_sample_pairs,
                size=(2, num_sample_pairs),
                device=device
            )

            # Make sure the first indices is smaller than the second indices.
            sampled_indices = torch.where(sampled_indices[0] < sampled_indices[1], sampled_indices, sampled_indices[[1, 0]])

            # Remove duplicated pairs and the equal pairs whose first index is equal to the second index.
            sampled_indices = torch.unique(sampled_indices, sorted=True, dim=1)
            sampled_indices = sampled_indices[:, sampled_indices[0] != sampled_indices[1]]

            num_pairs = sampled_indices.shape[1]

            # Initialize similarity tensor with IGNORE_TOKEN
            similarity_by_patch = torch.full(
                size=(bsz, 2 * num_pairs),
                fill_value=IGNORE_TOKEN,
                dtype=hidden_states.dtype,
                device=device
            )
            
            # Get original token indices for first and second tokens
            first_indices = non_text_indices[sampled_indices[0]]
            second_indices = non_text_indices[sampled_indices[1]]

            # Create token_index_by_patch by interleaving first and second indices
            token_index_by_patch = torch.stack([first_indices, second_indices], dim=1).reshape(1, -1)

            # Compute similarity for all pairs at once
            similarities = cosine_similarity(
                hidden_states[
                    torch.arange(bsz, device=device), 
                    first_indices, 
                    :
                ],
                hidden_states[
                    torch.arange(bsz, device=device), 
                    second_indices, 
                    :
                ],
            )

            # Assign similarities to second tokens of each pair (odd indices)
            similarity_by_patch[0, 1::2] = similarities
        else:
            # If not enough tokens, create empty token_index_by_patch
            token_index_by_patch = torch.zeros(bsz, 2 * num_sample_pairs, device=device)

        assert similarity_by_patch.shape[1] == token_index_by_patch.shape[1]
        return similarity_by_patch, token_index_by_patch

class FrameFusionRank(FrameFusion):
    def __init__(self, merging_ratio_list=None):
        """
        Initialize FrameFusionRank with a list of merging ratios for each layer.

        Args:
            merging_ratio_list (List[float]): List of merging ratios for each layer.
        """        
        super(FrameFusionRank, self).__init__(cost=0.0, similarity_lower_bound=0.0, ratio_lower_bound=0.0)
        # Initialize merging ratio list
        if merging_ratio_list is None:
            merging_ratio_list = [0.0]
        self.merging_ratio_list = list(merging_ratio_list)
        self.current_layer = 0

    def prepare(self, patch_type, patch_num, image_token_start_index, image_token_end_index, image_token_length, original_length, finish_merging=False, finish_pruning=False):
        """
        Prepare the module for the next layer.
        """
        super().prepare(patch_type, patch_num, image_token_start_index, image_token_end_index, image_token_length, original_length, finish_merging, finish_pruning)
        self.current_layer = 0

    def forward(self, hidden_states, position_embeddings, attention_mask, self_attn_weights = None):
        """
        Forward pass that uses layer-specific pruning ratios.

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
        if q_len > 1 and (not self.finish_merging):
            # align devices
            self.patch_type = self.patch_type.to(device)

            # Get similarities and indices
            similarity_by_patch, token_index_by_patch = self.compute_similarity_and_token_index_by_patch(hidden_states, self.patch_type, self.patch_num)
            
            frame_token_num = torch.sum(self.patch_type != TEXT_TOKEN).item()
            
            # Sort similarities and take top k based on current layer's merging ratio
            current_merging_ratio = self.merging_ratio_list[self.current_layer]
            k = int(current_merging_ratio * frame_token_num)
            if k > 0:
                topk_values, topk_indices = torch.topk(similarity_by_patch, k)
                topk_indices, _ = torch.sort(topk_indices)
                merge_index_by_patch = topk_indices[0]
                
                hidden_states, token_mask = self.merge_tokens_and_get_mask(hidden_states, similarity_by_patch, token_index_by_patch, merge_index_by_patch)
                # here only bsz=1
                # update patch type
                self.patch_type = self.patch_type.to(device)[token_mask].reshape(bsz, -1)
                hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, hidden_size)
                position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
                position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:,:,token_mask[0],:][:,:,:,token_mask[0]]
            
            self.current_layer += 1

            if self.current_layer >= len(self.merging_ratio_list):
                self.finish_merging = True
                self.finish_pruning = True # noqa: do not prune after merging

        return hidden_states, position_embeddings, attention_mask