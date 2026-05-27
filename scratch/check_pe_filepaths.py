import pickle
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

nodes_store_path = r"d:\AI HYBRID\index_store\nodes_store.pkl"

with open(nodes_store_path, "rb") as f:
    nodes = pickle.load(f)

for node in nodes:
    meta = getattr(node, "metadata", {})
    cname = meta.get("course_name", "")
    dept_short = meta.get("dept_short", "")
    filepath = meta.get("filepath", "")
    
    if "體育" in cname and dept_short == "未知系":
        print(f"Course: {cname} | Path: {filepath}")
        # 只看前 10 個
        break
