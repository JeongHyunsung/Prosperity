# Todo : Recursively inspect pkl file in data folder, 
# It is predicted to have information of activation function and weights
# for specific dataset - model combinations.

import argparse 
import pickle
import sys 
from typing import Any, Set

import numpy as np
import torch 

def is_torch_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)

def is_numpy_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)

def brief(x: Any) -> str:
    if is_torch_tensor(x):
        s = f"torch.Tensor shape={tuple(x.shape)} dtype={getattr(x, 'dtype', None)}"
        dev = getattr(x, "device", None)
        if dev is not None:
            s += f" device={dev}"
        return s

    if is_numpy_array(x):
        return f"np.ndarray shape={x.shape} dtype={x.dtype}"

    if isinstance(x, dict):
        return f"dict(len={len(x)})"

    if isinstance(x, (list, tuple, set, frozenset)):
        return f"{type(x).__name__}(len={len(x)})"

    if isinstance(x, (str, bytes, bytearray)):
        return f"{type(x).__name__}(len={len(x)})"

    if isinstance(x, (int, float, bool)) or x is None:
        return f"{type(x).__name__}(value={x})"

    return type(x).__name__


def safe_repr(x: Any, limit: int = 120) -> str:
    try:
        s = repr(x)
    except Exception:
        return ""
    return s if len(s) <= limit else s[:limit] + "…"

def sorted_keys(d: dict) -> list:
    keys = list(d.keys())
    try:
        keys = sorted(keys, key=lambda x:str(x))
    except Exception:
        pass
    return keys

def walk(
    x: Any,
    path: str,
    depth: int,
    max_depth: int,
    max_items: int,
    show_values: bool,
    visited: Set[int]
) -> None:
    pad = "  " * depth
    obj_id = id(x)
    if(obj_id in visited):
        print(f"{pad}{path}: (Already visited)")
        return
    visited.add(obj_id)

    if max_depth >= 0 and depth > max_depth:
        print(f"{pad}{path}: (Max depth reached)")
        return
    
    line = f"{pad}{path}: {brief(x)}"
    if show_values and isinstance(x, (str, int, float, bool)) or x is None:
        line += f"  {safe_repr(x)}"
    print(line)

    if is_torch_tensor(x):
        #print(x)
        return
    if is_numpy_array(x):
        #print(x)
        return

    if isinstance(x, dict):
        keys = x.keys()
        shown = keys[:max_items] if max_items >= 0 else keys
        rest = len(keys) - len(shown)

        for k in shown:
            walk(x[k], f"{path}.{k}", depth+1, max_depth, max_items, show_values, visited)
        if rest > 0:
            print(f"{pad}  ... and {rest} more keys")
    
    if isinstance(x, (set, frozenset)):
        elems = list(x)
        m = min(len(elems), max_items) if max_items > 0 else len(elems)
        for i in range(m):
            walk(elems[i], f"{path}[{i}]", depth+1, max_depth, max_items, show_values, visited)
        if len(elems) > m:
            print(f"{pad}  ... and {len(elems) - m} more items")
    
    if hasattr(x, "__dict__") and isinstance(getattr(x, "__dict__", None), dict):
        d = getattr(x, "__dict__")
        if d:
            walk(d, f"{path}.__dict__", depth+1, max_depth, max_items, show_values, visited)



def main() -> int:
    ap = argparse.ArgumentParser(description = "Inspect pkl file format")
    ap.add_argument("pkl_path")
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--max-items", type=int, default=-1)
    ap.add_argument("--show-values", action="store_true")
    args = ap.parse_args()

    with open(args.pkl_path, "rb") as f:
        obj = pickle.load(f)
    
    visited: Set[int] = set()
    walk(
        obj,
        path = 'root',
        depth = 0,
        max_depth = args.max_depth,
        max_items = args.max_items,
        show_values = args.show_values,
        visited = visited
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
