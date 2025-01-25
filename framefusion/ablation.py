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