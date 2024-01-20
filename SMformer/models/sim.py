from einops import (
    rearrange,
)
from torch import (
    Tensor,
)
from torch import (
    randn,
    zeros,
)


def get_indices_to_mask(
        batch_size: int,
        n_tokens: int,
        n_masked_tokens: int,
        device: str = 'cuda',
) -> Tensor:
    """
    Gets a set of indices per row for masking

    Args:
        batch_size (int): Batch size
        n_tokens (int): Number of tokens per row
        n_masked_tokens (int): Number of tokens to mask per row
        device (str): Desired device for the indices.
        Default is 'cuda'

    Returns (Tensor): Set of indices per row for masking
    """
    indices_to_mask = randn(batch_size, n_tokens, device=device)
    indices_to_mask = indices_to_mask.topk(
        k=n_masked_tokens,
        dim=1,
    )
    indices_to_mask = indices_to_mask.indices
    return indices_to_mask


def get_bitmask(
        batch_size: int,
        n_tokens: int,
        n_masked_tokens: int,
        device: str = 'cuda',
) -> Tensor:
    """
    Gets a bitmask for masking

    Args:
        batch_size (int): Batch size
        n_tokens (int): Number of tokens per row
        n_masked_tokens (int): Number of tokens to mask per row
        device (str): Desired device for the bitmask.
        Default is 'cuda'

    Returns (Tensor): Boolean tensor with True corresponding to masking
    the associated token
    """
    indices_to_mask = get_indices_to_mask(
        batch_size=batch_size,
        n_tokens=n_tokens,
        n_masked_tokens=n_masked_tokens,
        device=device,
    )

    bitmask = zeros(batch_size, n_tokens, device=device)
    bitmask = bitmask.scatter(
        dim=1,
        index=indices_to_mask,
        value=1,
    )
    bitmask = bitmask.bool()
    return bitmask


def do_mask_tokens(
        tokens: Tensor,
        mask_tokens: Tensor,
        bitmask: Tensor,
) -> Tensor:
    """
    Masks the tokens with a mask token given a bitmask

    Args:
        tokens (Tensor): Tokens to mask
        mask_tokens (Tensor): Tensor with the same shape as tokens filled with
        a mask token
        bitmask (Tensor): Bitmask for masking

    Returns (Tensor): The tokens masked with mask_tokens where bitmask is
    True
    """
    #遮的没问题
    bitmask = bitmask.unsqueeze(2)
    tokens = (~bitmask) * tokens + bitmask * mask_tokens
    return tokens


def get_patches(
        input: Tensor,
        patch_height: int,
        patch_width: int,
) -> Tensor:
    """
    Gets patches from input

    Args:
        input (Tensor): Input
        patch_height (int): Patch height
        patch_width (int): Patch width

    Returns (Tensor): Patches of the input
    """
    pattern = (
        'batch_size n_channels (n_patches_height patch_height) (n_patches_width patch_width) -> '
        'batch_size (n_patches_height n_patches_width) (n_channels patch_height patch_width)'
    )

    patches = rearrange(
        tensor=input,
        pattern=pattern,
        patch_height=patch_height,
        patch_width=patch_width,
    )
    return patches


def get_masked_patches_original(
        input: Tensor,
        patch_height: int,
        patch_width: int,
        bitmask: Tensor,
) -> Tensor:
    """
    Gets patches from input that are supposed to be masked

    Args:
        input (Tensor): Input to extract patches from
        patch_height (int): Patch height
        patch_width (int): Patch width
        bitmask (Tensor): Bitmask that was used for masking

    Returns (Tensor): Original version of the patches that are supposed
    to be masked
    """
    patches = get_patches(
        input=input,
        patch_height=patch_height,
        patch_width=patch_width,
    )
    maskes_patches_original = patches[bitmask]
    return maskes_patches_original
