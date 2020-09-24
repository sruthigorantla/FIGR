import pandas as pd
import numpy as np

from figr.utils import swap, get_next_candidate


def figr(data_id, id_2_protected, NUM_GROUPS, ALPHA, p_deviation, K):

    NUM_ELEMENTS = len(data_id)
    ALPHA = max(ALPHA - p_deviation, 1.0/NUM_GROUPS + 0.01) 
    EPSILON =  (2.0/K)*(1+1.0/(ALPHA  - 1.0/NUM_GROUPS))
        
    
    BLOCK_SIZE = int(EPSILON * K * 0.5) 
    UPPER_BOUND = int(ALPHA * BLOCK_SIZE) 
    
    target_data = []
    for idx in range(0, NUM_ELEMENTS, UPPER_BOUND):

        target_block = [None] * BLOCK_SIZE
        target_block[:UPPER_BOUND] = data_id[idx:idx+UPPER_BOUND]
        target_data.extend(target_block)

    # Loop across blocks
    for idx in range(0, len(target_data), BLOCK_SIZE):

        START_ID = idx
        END_ID = idx+BLOCK_SIZE-1

        # Looping inside each block
        for curr_id in range(START_ID, END_ID+1):
  
            if (curr_id < len(target_data)) and (target_data[curr_id] is None):
                
                candidate_id = curr_id + 1

                # Try and fill the current None value from the remaining ids as per fairness
                while (candidate_id < len(target_data)): 
                    # Search for next non-empty ID
                    candidate_id = get_next_candidate(target_data, start = candidate_id) 

                    # if next candidate is not found
                    if candidate_id == -1:
                        break

                    swap(target_data, curr_id, candidate_id) # Potential swap
                    
                    if is_block_fair(target_data[START_ID:END_ID+1], id_2_protected, UPPER_BOUND, NUM_GROUPS):
                        # Swap confirmed
                        break
                    else:
                        # Swap declined (reverse)
                        swap(target_data, curr_id, candidate_id)
                    
                    candidate_id += 1 

    final_rank = [] 
    for item in target_data:
        if item is not None:
            final_rank.append(item)
    
    return final_rank

def is_block_fair(block, id_2_protected, UPPER_BOUND, NUM_GROUPS):

    # check if the group fairness constraint in the block is satisfied
    PROTECTED = np.zeros(NUM_GROUPS)
    for item in block:
        if item is not None:
            PROTECTED[id_2_protected[item]] += 1
    return (PROTECTED <= UPPER_BOUND).all()