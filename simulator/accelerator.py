from weight_dependent_cache import ReplacementPolicy

class Accelerator:
    def __init__(self, type, adder_array_size, LIF_array_size, tile_size_M, tile_size_K, product_sparsity=True, dense=False, issue_type=2, mem_if_width=1024, use_global_cache=False, cache_policy='max_nnz', min_th=1, max_th=5, m_multiple=1):
        self.type = type
        self.num_popcnt = 8
        self.adder_array_size = adder_array_size  # tile size N
        self.LIF_array_size = LIF_array_size
        self.multiplier_array_size = 32
        self.num_exp = 8
        self.num_div = 1
        self.SpMM_tile_size_M = tile_size_M
        self.SpMM_tile_size_K = tile_size_K
        self.mem_if_width = mem_if_width
        self.tech_node = 0.028
        self.sram_size = {}
        self.sram_size['wgt'] = self.SpMM_tile_size_K * self.adder_array_size * 8    # global buffer 16 * 128 * 8 bit
        self.sram_size['act'] = self.SpMM_tile_size_M * self.SpMM_tile_size_K * 1    # global buffer 16 * 256 bit
        self.sram_size['out'] = self.SpMM_tile_size_M * self.adder_array_size * 8       # global buffer 258 * 128 * 8 bit

        self.product_sparsity = product_sparsity
        self.dense = dense
        self.issue_type = issue_type # 1: search over tree through prefix 2: sorting based issue

        self.use_global_cache = use_global_cache # True : use global weight dependent cache
        policy_map = {
            'max_nnz': ReplacementPolicy.MAX_NNZ,
            'lru': ReplacementPolicy.LRU,
            'fifo': ReplacementPolicy.FIFO,
            'random': ReplacementPolicy.RANDOM,
        }
        if self.use_global_cache:
            self.min_th = min_th
            self.max_th = max_th
            self.cache_policy = policy_map[cache_policy]
            self.m_multiple = m_multiple
