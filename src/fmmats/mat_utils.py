from typing import Union
import pandas as _pd
import math as _math
import numpy as _np

def merge_blocks(
    mapping: _pd.DataFrame, 
    blocks: 'dict[str, _pd.DataFrame]') -> _pd.DataFrame:

    design_mat = mapping
    for block_name, block in blocks.items():
        design_mat = design_mat.join(block, rsuffix=f'_{block_name}')
    
    return design_mat


def calc_num_batches(
    data: Union[_pd.DataFrame, _np.ndarray],
    batch_size: int = None) -> int:

    if batch_size is None:
        return 1
    return _math.ceil(len(data) / batch_size)


def fillna_across_blocks(
    blocks: 'dict[str, _pd.DataFrame]', 
    value: int = 0) -> _pd.DataFrame:

    return {block_name: block.fillna(value) for block_name, block in blocks.items()}