import os as _os
import shutil as _shutil
import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.datasets import dump_svmlight_file as _dump_svmlight_file

from .mat_utils import *


def write_mat(
    mapping_base: _pd.DataFrame, 
    blocks: 'dict[str, _pd.DataFrame]', 
    out_dir: str, 
    prefix: str = '', 
    seed: int = None, 
    block_groups: 'dict[str, list[int]]' = None,
    batch_size: int = None, 
    target_col: str = 'target',
    train_size: float = 0.75,
    test_size: float = 0.25) -> None:
    '''
    Left join the ``mapping_base`` with all of the ``blocks`` by index and then write the resulting design matrix in ``.libfm`` format.
    
    ### Parameters:
        mapping_base : DataFrame
            The main DataFrame that indicates the relational mappings of tables using its indices.

        blocks : dict[str, DataFrame]
            The relational blocks to join with the ``mapping_base`` by indices.

        out_dir : str
            The directory to write the resulting ``.libfm`` design matrix.

        prefix : str, default=''
            The prefix to add to the filename output.

        seed : int, default=None
            The seed to use for rng.

        block_groups : dict[str, list[int]], default=None
            The grouping of columns to use for more advanced regularization. Groupings for each block should start at 0 and increment up to n-1 groups.

        batch_size : int, default=None
            The number of ``mapping_base`` rows to join at a time. Use when the relationally joined matrix is too large to store in memory all at once.

        target_col : str, default="target"
            The name of the column in ``mapping_base`` to use as the target column.

        train_size : float or int, default=0.75
            The proportion of data to use for the train set.

        test_size : float or int, default=0.25
            The proportion of data to use for the test set.
    '''

    blocks = fillna_across_blocks(blocks, 0)

    mapping_split = {}
    mapping_split['train'], mapping_split['test'] = _train_test_split(mapping_base, train_size=train_size, test_size=test_size, random_state=seed)

    # Write design matrix to 'out_dir' in batches saved temporarily in '.tmp_libfm_mats'
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
    
    # If 'block_groups' was provided, generate the grouping, distinguishing between each block's grouping.
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
    prefix: str = '', 
    seed: int = None, 
    block_groups: 'dict[str, list[int]]' = None,
    target_col: str = 'target',
    train_size: float = 0.75,
    test_size: float = 0.25) -> None:
    '''
    Write each block in ``blocks`` as a relational block for LibFM's block structure format.
    
    ### Parameters:
        mapping_base : DataFrame
            The main DataFrame that indicates the relational mappings of tables using its indices.

        blocks : dict[str, DataFrame]
            The relational blocks to join with the ``mapping_base`` by indices.

        out_dir : str
            The directory to write the resulting ``.libfm`` design matrix.

        prefix : str, default=''
            The prefix to add to the filename output.

        seed : int, default=None
            The seed to use for rng.

        block_groups : dict[str, list[int]], default=None
            The grouping of columns to use for more advanced regularization. Groupings for each block should start at 0 and increment up to n-1 groups.

        target_col : str, default="target"
            The name of the column in ``mapping_base`` to use as the target column.

        train_size : float or int, default=0.75
            The proportion of data to use for the train set.

        test_size : float or int, default=0.25
            The proportion of data to use for the test set.
    '''

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
            block_mappings = mapping.droplevel(level=mapping.index.names.difference(block.index.names)).join(block.droplevel(level=block.index.names.difference(mapping.index.names)), rsuffix=f'_{block_name}').block_row_num

            _np.savetxt(_os.path.join(out_dir, f'{prefix}{block_name}.{split}'), block_mappings, fmt='%i')
    
    if bool(block_groups):
        for block_name, block_group in block_groups.items():
            _np.savetxt(_os.path.join(out_dir, f'{prefix}{block_name}.groups'), block_group, fmt='%i')
