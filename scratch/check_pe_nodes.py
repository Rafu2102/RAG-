import pickle
import os
import sys

# 強制標準輸出為 utf-8
sys.stdout.reconfigure(encoding='utf-8')

nodes_store_path = r"d:\AI HYBRID\index_store\nodes_store.pkl"

if not os.path.exists(nodes_store_path):
    print("Nodes store not found!")
    exit(1)

with open(nodes_store_path, "rb") as f:
    nodes = pickle.load(f)

print(f"Total nodes loaded: {len(nodes)}")

pe_nodes = []
for node in nodes:
    meta = getattr(node, "metadata", {})
    cname = meta.get("course_name", "")
    if "體育" in cname:
        pe_nodes.append(meta)

print(f"Total PE nodes: {len(pe_nodes)}")

# 印出不同的體育課 metadata 屬性
seen = set()
count = 0
for meta in pe_nodes:
    cname = meta.get("course_name", "")
    dept_short = meta.get("dept_short", "")
    dept_full = meta.get("dept_full", "")
    dept_raw = meta.get("department", "")
    teacher = meta.get("teacher", "")
    # 檔案路徑也看一下
    filepath = meta.get("filepath", "")
    key = (cname, dept_short, dept_full, dept_raw, teacher)
    if key not in seen:
        seen.add(key)
        print(f"Course: {cname} | DeptShort: {dept_short} | DeptFull: {dept_full} | RawDept: {dept_raw} | Teacher: {teacher} | Path: {filepath}")
        count += 1
        if count >= 40:
            break
