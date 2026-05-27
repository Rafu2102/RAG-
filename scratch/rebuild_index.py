# -*- coding: utf-8 -*-
import os
import sys

# 強制將 stdout 與 stderr 重建為 utf-8 以防 Windows cp950 編碼出錯
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from rag.index_manager import load_and_index

def rebuild():
    print("[Rebuild] Start rebuilding all indices (FAISS + BM25)...")
    config.setup_logging()
    
    # 強制重新建立索引
    nodes, faiss_idx, bm25_idx = load_and_index(force_rebuild=True)
    
    print(f"\n[Summary] Rebuild Complete!")
    print(f"   Total Nodes count: {len(nodes)}")
    print(f"   Vector dimension: {faiss_idx.d}")
    print(f"   Vector count: {faiss_idx.ntotal}")

if __name__ == "__main__":
    rebuild()
