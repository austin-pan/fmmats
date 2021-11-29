from typing import Union as _Union
import pandas as _pd
import math as _math
import numpy as _np

def merge_blocks(
    mapping: _pd.DataFrame, 
    blocks: 'dict[str, _pd.DataFrame]') -> _pd.DataFrame:
    '''Merges all blocks with the mapping by index using a left-join and returns the resulting dataframe.'''

    design_mat = mapping
    for block_name, block in blocks.items():
        design_mat = design_mat.join(block, rsuffix=f'_{block_name}')
    
    return design_mat


def calc_num_batches(
    data: _Union[_pd.DataFrame, _np.ndarray],
    batch_size: int = None) -> int:
    '''Returns the number of batches to use given a dataset and the batch size to use.'''

    if batch_size is None:
        return 1
    return _math.ceil(len(data) / batch_size)


def fillna_across_blocks(
    blocks: 'dict[str, _pd.DataFrame]', 
    value: int = 0) -> _pd.DataFrame:
    '''Returns the blocks passed in after filling the NaN values with ``value``.'''

    return {block_name: block.fillna(value) for block_name, block in blocks.items()}


def normalize_columns(
    df: _pd.DataFrame,
    in_place: bool = False) -> _pd.DataFrame:
    '''
    Returns the normalized by column version of the dataframe using min-max normalization to achieve a range of [0-1].
    Columns filled with the same value will be replaced with ``NaN``.
    '''
    
    if not in_place:
        df = df.copy()

    df_min = df.min()
    df_max = df.max()
    df = (df - df_min) / (df_max - df_min) # Normalize columns to a range of [0-1]
    
    return df


def standardize_columns(
    df: _pd.DataFrame,
    in_place: bool = False) -> _pd.DataFrame:
    '''
    Returns the normalized by column version of the dataframe using min-max normalization to achieve a mean of 0 and std of 1.
    Columns filled with the same value will be replaced with ``NaN``.
    '''
    
    if not in_place:
        df = df.copy()

    df_mean = df.mean()
    df_std = df.std()
    df = (df - df_mean) / df_std # Standardize columns to have a mean of 0 and std of 1
    
    return df