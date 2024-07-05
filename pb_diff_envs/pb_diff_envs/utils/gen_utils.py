import numpy as np

def check_data_len(data, tgt_len):
    for k,v in data.items():
        assert len(v) == tgt_len, f'{k}: {len(v)}'

def npify(data):
    for k in data:
        if k == 'eval_sols':
            continue
        if k in ['terminals', 'pos_reset', 'is_sol_kp', 'is_suc']:
            dtype = np.bool_
        elif k == 'maze_idx':
            dtype = np.int32
        else:
            dtype = np.float32

        # data[k] = np.array(data[k], dtype=dtype)
        data[k] = np.concatenate(data[k], dtype=dtype)
