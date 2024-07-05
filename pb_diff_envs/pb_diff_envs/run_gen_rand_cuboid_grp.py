import gym, pdb
import numpy as np
import os
'''
This file used to generate the wall location list and save it to npy.
After that, we can directly load the list by setting gen_data=False.
Generally, you will need to run this script after you register a new env.
'''

if __name__ == '__main__':
    import pb_diff_envs.environment
    e_list  = []
    
    ## ---------- TestOnly ----------
    # e_list.append("randSmaze2d-testOnly-v0")
    # e_list.append("kuka7d-testOnly-v8")
    # e_list.append("dualkuka14d-testOnly-v9")
    # e_list.append("randSmaze2d-C43-testOnly-v0")
    ## ------------------------------


    ## ----------- Kuka -------------
    # e_list.append("dualkuka14d-ng25hs25k-rand-nv5-se1005-vr0702-hExt22-v0")
    # e_list.append("kuka7d-ng2ks25k-rand-nv4-se0505-vr0202-hExt20-v0")
    ## ------------------------------


    ## ------- Composed Maze2D ----------
    # e_list.append( 'ComprandSmaze2d-ms55nw6nw1-hExt0505-gsdiff-v0' )
    # e_list.append( 'ComprandSmaze2d-ms55nw6nw2-hExt0505-gsdiff-v0' )
    # e_list.append( 'ComprandSmaze2d-ms55nw6nw3-hExt0505-gsdiff-v0' )
    # e_list.append( 'ComprandSmaze2d-ms55nw6nw4-hExt0505-gsdiff-v0' )
    # e_list.append( 'ComprandSmaze2d-ms55nw6nw5-hExt0505-gsdiff-v0' )
    # e_list.append( 'ComprandSmaze2d-ms55nw6nw6-hExt0505-gsdiff-v0' )
    ## ----------------------------------

    ## ------- Dyn Maze2D --------
    # e_list.append( 'DynrandSmaze2d-testOnly-v2' )
    # e_list.append( 'DynrandSmaze2d-ng1ks10k-ms55nw1-hExt10-v0' )

    ## --- Maze2D ---
    e_list.append( "randSmaze2d-ng3ks25k-ms55nw6-hExt05-v0" )
    # e_list.append( "randSmaze2d-ng3ks35k-ms55nw3-hExt07-v0" )
    # e_list.append( "randSmaze2d-C43-ng3ks25k-ms55nw21-hExt05-v0" )


    for e_l in e_list:
        if ('Comp' not in e_l):
            ## for compositional envs, we only needs the training parts
            dvg_group = gym.make(e_l,
                            load_mazeEnv=False, gen_data=True, is_eval=False)
            ## dvg_group.recs_grp.save_center_and_mode()

        ## For Evaluation    
        dvg_group = gym.make(e_l,
                            load_mazeEnv=False, gen_data=True, is_eval=True)
    