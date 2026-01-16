import torch
from weight_dependent_cache import GlobalCache

def run_scenario(rows, m=8, k=None, realloc=5, addition=1):
    """
    rows: list of list/tuple of 0/1
    """
    if k is None:
        k = len(rows[0])
    cache = GlobalCache(n=1, m=m, k=k,
                        reallocation_threshold=realloc,
                        addition_threshold=addition)

    act = torch.tensor(rows, dtype=torch.bool)
    pre, pref = cache.preprocess_tensor(act)

    print("=== preprocess_tensor result ===")
    for i in range(act.shape[0]):
        print(f"[{i}] in : {act[i].tolist()}  -> out: {pre[i].tolist()}  prefix: {int(pref[i])}")

    print("\ncache_keys after preprocess:")
    for i in range(cache.m):
        print(f"  slot {i}: valid={bool(cache.valid_bits[i])} key={cache.cache_keys[i].tolist()} nnz={int(cache.cache_nnz[i])}")

if __name__ == "__main__":
    # 원하는 패턴을 여기에 넣고 실행하세요
    rows = [
        [1, 0, 1, 0],
        [1, 0, 1, 1],  # superset of row0
        [0, 1, 0, 1],
        [1, 1, 0, 0],  # another candidate
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],  # superset of row0
        [0, 1, 0, 1],
        [1, 1, 0, 0],  # another candidate
        [1, 1, 1, 1]
    ]
    run_scenario(rows)