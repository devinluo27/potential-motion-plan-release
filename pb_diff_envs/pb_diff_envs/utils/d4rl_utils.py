from functools import reduce
from operator import mul

'''
compute index in return of combinations
from https://stackoverflow.com/questions/14455634/find-the-index-of-a-given-combination-of-natural-numbers-among-those-returned
'''
def nck_safe(n, k):
    if k < 0 or n < k: return 0
    return reduce(mul, range(n, n-k, -1), 1) // reduce(mul, range(1, k+1), 1)

def find_idx(comb,n):
    '''remember n need to plus one before pass in, to fit the math'''
    k=len(comb)
    b=bin(sum(1<<(x-1) for x in comb))[2:]
    idx=0
    for s in b[::-1]:
        if s=='0':
            idx+=nck_safe(n-2,k-1)
        else:
            k-=1
        n-=1
    return idx
