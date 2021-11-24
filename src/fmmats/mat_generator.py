import os as _os
import shutil as _shutil
import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.datasets import dump_svmlight_file as _dump_svmlight_file

from .mat_utils import *


def write_mats(
    mapping_base: _pd.DataFrame, 
    blocks: 'dict[str, _pd.DataFrame]', 
    out_dir: str, 
    prefix: str = '', 
    seed: int = None, 
    block_groups: 'dict[str, any]' = None,
    batch_size: int = None, 
    target_col: str = 'target',
    train_size: float = 0.75,
    test_size: float = 0.25) -> None:

    blocks = fillna_across_blocks(blocks, 0)

    mapping_split = {}
    mapping_split['train'], mapping_split['test'] = _train_test_split(mapping_base, train_size=train_size, test_size=test_size, random_state=seed)

    tmp_dir = f'.tmp_libfm_mats'
    for split in mapping_split:
        mapping = mapping_split[split]

        if not _os.path.isdir(tmp_dir):
            _os.mkdir(tmp_dir)
        else:
            for f in _os.listdir(tmp_dir):
                _os.remove(_os.path.join(tmp_dir, f))

        num_batches = calc_num_batches(mapping, batch_size)
        for i, mapping_batch in enumerate(_np.array_split(mapping, num_batches)):
            batch = merge_blocks(mapping_batch, blocks)

            _dump_svmlight_file(batch.drop(columns=target_col), _np.ravel(batch[[target_col]]), _os.path.join(tmp_dir, f'batch_{i}.libfm'))
    
        with open(_os.path.join(out_dir, f'{prefix}{split}.libfm'), 'wb') as wfd:
            for f in _os.listdir(tmp_dir):
                with open(_os.path.join(tmp_dir, f), 'rb') as fd:
                    _shutil.copyfileobj(fd, wfd)

    if _os.path.isdir(tmp_dir):
        for f in _os.listdir(tmp_dir):
            _os.remove(_os.path.join(tmp_dir, f))
        _os.rmdir(tmp_dir)
    
    if bool(block_groups):
        init = 0
        for block_name in block_groups:
            block_group = _np.array(block_groups[block_name]) + init
            block_groups[block_name] = block_group
            init = block_group.max() + 1
        groups = _np.concatenate(list(block_groups.values()))

        _np.savetxt(_os.path.join(out_dir, f'{prefix}_groups'), groups, fmt='%i')


def write_bs_mats(
    mapping_base: _pd.DataFrame, 
    blocks: 'dict[str, _pd.DataFrame]', 
    out_dir: str, 
    block_groups: 'dict[str, list[int]]' = None,
    prefix: str = '', 
    seed: int = None, 
    target_col: str = 'target',
    train_size: float = 0.75,
    test_size: float = 0.25) -> None:

    if bool(block_groups) and block_groups.keys() != blocks.keys():
        raise ValueError('{block_groups} keys have to match {blocks} keys.')
    
    blocks = fillna_across_blocks(blocks, 0)

    for block_name, block in blocks.items():
        _dump_svmlight_file(block, _np.zeros(block.shape[0]), f'{out_dir}/{prefix}{block_name}.libfm')

    mapping_split: dict[str, _pd.DataFrame] = {}
    mapping_split['train'], mapping_split['test'] = _train_test_split(mapping_base, train_size=train_size, test_size=test_size, random_state=seed)

    for split in mapping_split:
        mapping = mapping_split[split]
        _np.savetxt(_os.path.join(out_dir, f'{prefix}{split}'), _np.ravel(mapping[target_col]), fmt='%f')

        mapping = mapping.drop(columns=mapping.columns)
        for block_name, block in blocks.items():
            block = block.drop(columns=block.columns)
            block['block_row_num'] = range(len(block))
            block_mappings = mapping.droplevel(level=mapping.index.names.difference(block.index.names)).join(block.droplevel(level=block.index.names.difference(mapping.index.names)), lsuffix=f'_{block_name}').block_row_num

            _np.savetxt(_os.path.join(out_dir, f'{prefix}{block_name}.{split}'), block_mappings, fmt='%i')
    
    if bool(block_groups):
        for block_name, block_group in block_groups.items():
            _np.savetxt(_os.path.join(out_dir, f'{prefix}{block_name}.groups'), block_group, fmt='%i')

