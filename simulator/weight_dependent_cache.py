import torch
from enum import Enum

class ReplacementPolicy(Enum):
    MAX_NNZ = "max_nnz"
    LRU = "lru" 
    FIFO = "fifo" # queue
    RANDOM = "random" 

class GlobalCache:
    def __init__(self, n, m, k, addition_threshold, reallocation_threshold, replacement_policy: ReplacementPolicy):
        self.n = n
        self.m = m
        self.k = k
        self.cache_keys = torch.zeros((m, k), dtype=torch.bool)
        self.valid_bits = torch.zeros((m), dtype=torch.bool)
        self.cache_nnz = torch.zeros((m), dtype=torch.int32)
        self.addition_threshold = addition_threshold
        self.reallocate_threshold = reallocation_threshold
        self.replacement_policy = replacement_policy

        self.fifo_elim_idx = 0
        self.lru_approx = torch.zeros((m), dtype=torch.int32) # for LRU approximation
        self.lifetime_counter = 0


        
    def miss_handle(self, row : torch.Tensor):
        # row : 1 x k vector (missed row) 
        # returns : None (updates cache)

        ## How to handle miss case? (fully associative cache replacement policy : ideal assumption)
        ### 1. find an invalid entry -> allocate
        ### 2. if all valid, find nnz >= threshold -> reallocate 
        ### 3. if all valid and all nnz < threshold -> do nothing (not cached)

        row_nnz = torch.sum(row != 0).item()

        if not torch.all(self.valid_bits):
            invalid_idx = torch.argmin(self.valid_bits.int()).item()
            self.cache_keys[invalid_idx] = row
            self.valid_bits[invalid_idx] = True
            self.cache_nnz[invalid_idx] = row_nnz

            if self.replacement_policy == ReplacementPolicy.LRU:
                self.lru_approx[invalid_idx] = 1
            return

        if self.replacement_policy == ReplacementPolicy.MAX_NNZ:
            evictable = self.cache_nnz >= self.reallocate_threshold
            if torch.sum(evictable) == 0:
                return 
            
            evictable_nnz = torch.where(evictable, self.cache_nnz, torch.tensor(-1))
            max_nnz_idx = torch.argmax(evictable_nnz).item()
            if row_nnz < evictable_nnz[max_nnz_idx].item():
                self.cache_keys[max_nnz_idx] = row
                self.cache_nnz[max_nnz_idx] = row_nnz
                return
        
        elif self.replacement_policy == ReplacementPolicy.RANDOM:
            eviction_idx = torch.randint(0, self.m, (1,)).item()
            self.cache_keys[eviction_idx] = row
            self.cache_nnz[eviction_idx] = row_nnz
            return
        
        elif self.replacement_policy == ReplacementPolicy.FIFO:
            eviction_idx = self.fifo_elim_idx
            self.cache_keys[eviction_idx] = row
            self.cache_nnz[eviction_idx] = row_nnz
            self.fifo_elim_idx = (self.fifo_elim_idx + 1) % self.m
            return
        
        elif self.replacement_policy == ReplacementPolicy.LRU:
            evictable = self.lru_approx == 0
            if torch.sum(evictable) == 0:
                return
            evictable_indices = torch.nonzero(evictable).flatten()
            eviction_idx = evictable_indices[0].item()
            self.cache_keys[eviction_idx] = row
            self.cache_nnz[eviction_idx] = row_nnz
            self.lru_approx[eviction_idx] = 1
            return
        else:
            raise NotImplementedError("Replacement policy not implemented.")
        return 
            

    def lookup_rows(self, row : torch.Tensor): 
        # row : 1 X k vector (current activation row)
        # returns : 1 X k vector (product sparse activation row (after preprocessing)) 

        
        and_result = torch.logical_and(row, self.cache_keys)
        equalities = torch.eq(and_result, self.cache_keys)
        is_equal = torch.all(equalities, dim=-1)
        is_equal = torch.logical_and(is_equal, self.valid_bits)

        if torch.sum(is_equal) == 0:
            
            self.miss_handle(row)
            return row, -1

        subset_row = self.cache_keys[is_equal]
        subset_nnz = torch.sum(subset_row, dim=-1)
        
        subset_index = torch.nonzero(is_equal).flatten()
        max_nnz, max_idx = torch.max(subset_nnz, dim=0)
        reused_prefix_index = subset_index[max_idx].item()

        processed_row = torch.logical_xor(row, self.cache_keys[reused_prefix_index])
        if torch.sum(processed_row != 0).item():
            self.miss_handle(row)

        return processed_row, reused_prefix_index

    def preprocess_tensor(self, act : torch.Tensor):
        # act : m X k tensor (current activation tile) 
        # returns A : m x k tensor (product sparse activation tile (after preprocessing)) 
        # returns B : m tensor (which rows are reused to compute each row index)

        
        preprocessed_act = act.clone()
        prefix_array = torch.ones(act.shape[0])
        prefix_array = -prefix_array # set all to -1
        
        for i in range(act.shape[0]):
            
            self.lifetime_counter += 1
            if(self.lifetime_counter % 1000 == 0):
                self.lru_approx = torch.zeros((self.m), dtype=torch.int32)

            cur_row = act[i]
            cur_nnz = torch.sum(cur_row != 0).item()
            if cur_nnz < self.addition_threshold+1: 
                if cur_nnz == self.addition_threshold:
                    self.miss_handle(cur_row)
                continue
            preprocessed_row, reused_prefix_index = self.lookup_rows(cur_row)
            if reused_prefix_index != -1:
                self.lru_approx[reused_prefix_index] = 1
                prefix_array[i] = reused_prefix_index
                preprocessed_act[i] = preprocessed_row
        return preprocessed_act, prefix_array

