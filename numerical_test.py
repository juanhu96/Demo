import numpy as np
import pandas as pd


def main():
    # Set the random seed for reproducibility
    np.random.seed(42)
    num_seeds = 5
    random_seed_list = np.random.randint(0, 100000, size=num_seeds)

    for random_seed in random_seed_list:
        test(random_seed)

    return


def test(random_seed):

    np.random.seed(random_seed)
    num_blocks = 40 # two block, three location
    weights = np.random.rand(num_blocks)
    weights /= weights.sum()
    u_0 = np.random.exponential(scale=2, size=num_blocks) # exponential utility of each block at location 0
    u_1 = np.random.exponential(scale=2, size=num_blocks) # exponential utility of each block at location 1
    u_2 = np.random.exponential(scale=2, size=num_blocks) # exponential utility of each block at location 2

    # weights = np.array([0.2, 0.8]) # population weights of each block
    # u_0 = np.array([2, 3]) # exponential utility of each block at location 0
    # u_1 = np.array([1, 4]) # exponential utility of each block at location 1

    ## Single location
    rho_0, rho_1, rho_2 = np.sum(weights * u_0/(1+u_0)), np.sum(weights * u_1/(1+u_1)), np.sum(weights * u_2/(1+u_2))
    y_0, y_1, y_2 = rho_0/(1-rho_0), rho_1/(1-rho_1), rho_2/(1-rho_2) # single location induced utility (what we're using)

    # Multi location
    rho_0, rho_1, rho_2 = 0,0,0
    for i in range(num_blocks):

        loc_0 = weights[i] * u_0[i] / (1 + u_0[i]) * u_0[i] / (u_0[i] + u_1[i] + u_2[i])
        loc_1 = weights[i] * u_1[i] / (1 + u_1[i]) * u_1[i] / (u_0[i] + u_1[i] + u_2[i])
        loc_2 = weights[i] * u_2[i] / (1 + u_2[i]) * u_2[i] / (u_0[i] + u_1[i] + u_2[i])

        rho_0 += loc_0
        rho_1 += loc_1
        rho_2 += loc_2

    blk_share = np.array([rho_0, rho_1, rho_2]) # actual 
    tct_share = np.array([y_0/(1+y_0) * y_0/(y_0 + y_1 + y_2), y_1/(1+y_1) * y_1/(y_0 + y_1 + y_2), y_2/(1+y_2) * y_2/(y_0 + y_1 + y_2)]) # what we are doing
    
    print(np.mean(np.abs((blk_share - tct_share) / blk_share)))
    
    return 


def test_matrix(weights, u_0, u_1):
    '''
    weights: population weight of each block, sum to 1
    u_0: exponential utility of each block at location 0
    u_1: exponential utility of each block at location 1
    '''

    num_blocks = len(weights)

    ## Single location
    rho_0, rho_1 = np.sum(weights * u_0/(1+u_0)), np.sum(weights * u_1/(1+u_1))
    y_0, y_1 = rho_0/(1-rho_0), rho_1/(1-rho_1)

    # Multi location
    rho_0, rho_1 = 0, 0
    for i in range(num_blocks):

        loc_0 = weights[i] * u_0[i] / (1 + u_0[i]) * u_0[i] / (u_0[i] + u_1[i])
        loc_1 = weights[i] * u_1[i] / (1 + u_1[i]) * u_1[i] / (u_0[i] + u_1[i])
        rho_0 += loc_0
        rho_1 += loc_1

    blk_share = np.array([rho_0, rho_1])
    tct_share = np.array([y_0/(1+y_0) * y_0/(y_0 + y_1 + y_2), y_1/(1+y_1) * y_1/(y_0 + y_1 + y_2), y_2/(1+y_2) * y_2/(y_0 + y_1 + y_2)])
    print(np.max(np.abs(blk_share-tct_share)))
    
    return 


if __name__ == "__main__":
    main()