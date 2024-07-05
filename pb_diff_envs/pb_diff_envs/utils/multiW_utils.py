import numpy as np


def pad_wall_hExt_single(wloc, hExt):
    '''
    Pad the size of the squared block to wloc, shape of last dim + 1,
    For single env.
    wloc: B nw 2
    hExt: (2,) e.g., (0.5, 0.5), not a range, just h,w
    '''
    ## only 1 hExt
    # hExt = el.hExt_range
    assert hExt[0] == hExt[1]
    h_shape = (wloc.shape[0], wloc.shape[1], 1)
    hExt = np.full( shape=h_shape, fill_value=hExt[0] ) # expand
    ## (B nw 2) cat (B nw 1) = B nw 3
    return np.concatenate( [wloc, hExt], axis=2 )


def pad_wall_hExt_comp(wloc, hExt_c, num_walls_c):
    '''
    Pad the size of the squared block to wloc, shape of last dim + 1,
    For composed env, only support 2 envs.
    wloc: B nw 2
    hExt_c: (2,2) e.g., [ [0.5, 0.5], [0.7],[0.7] ]
    '''
    ## will have 2 hExt
    # hExt_c = el.hExt_range_c
    assert (hExt_c[:, 0] == hExt_c[:, 1]).all()
    nw_c = num_walls_c

    assert len(nw_c) == 2 and wloc.shape[1] == nw_c.sum()

    h_shape_1 = (wloc.shape[0], nw_c[0], 1)
    hExt_1 = np.full( shape=h_shape_1, fill_value=hExt_c[0, 0] )

    h_shape_2 = (wloc.shape[0], nw_c[1], 1)
    hExt_2 = np.full( shape=h_shape_2, fill_value=hExt_c[1, 0] )

    hExt = np.concatenate([hExt_1, hExt_2], axis=1)
    # pdb.set_trace()
    return np.concatenate( [wloc, hExt], axis=2 )


def pad_num_walls(wloc, max_num_walls, pad_value=-1):
    '''
    In multiW, number of walls might be different, 
    Pad along nw dimension, pad to max number, pad -1 by default
    wloc: B, nw, 3 (remember we have one dim of hExt)
    '''
    # el_data: dict = self.el_data_list[i_e]
    # w = el_data['infos/wall_locations']
    assert wloc.shape[2] == 3
    n_pad = max_num_walls - wloc.shape[1] # minus n_w in the env

    ## use -1 to pad
    # w_pad = - np.ones(shape=(wloc.shape[0], n_pad, wloc.shape[2]))
    w_pad = np.full(shape=(wloc.shape[0], n_pad, wloc.shape[2]), fill_value=pad_value)

    return np.concatenate([wloc, w_pad], axis=1)

    # num_w = np.zeros(shape=(w.shape[0]), dtype=np.int32) + self.num_walls_list[i_e]
    # el_data['infos/num_walls'] = num_w



