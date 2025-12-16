import pickle, sys, types
from collections.abc import Mapping, Sequence

with open("/home/ws/navsim_workspace/dataset/traj_pdm_v2/ori/navtrain_8192.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))

# 若是 dict，看看 keys
if isinstance(obj, Mapping):
    print("num_keys:", obj.keys())
    print("keys (sample):", list(obj.keys())[:20])

# 若是 list/tuple，看看长度和前几个元素的类型
elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
    print("length:", len(obj[0]['anns']['instance_tokens']))
    print("first item type:", type(obj[0]) if obj else None)
    print("first item preview:", obj[0] if obj and len(str(obj[0])) < 500 else type(obj[0]))

# 其它常见类型
elif hasattr(obj, "shape"):
    print("array-like shape:", getattr(obj, "shape", None), "dtype:", getattr(obj, "dtype", None))
else:
    print(repr(obj)[:1000])