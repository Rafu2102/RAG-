# -*- coding: utf-8 -*-
import pickle
import os
import sys
import hashlib
import numpy as np
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
# 不需從 data_loader 導入快取路徑，因為已在下方定義

def init_cache():
    config.setup_logging()
    nodes_path = config.NODES_STORE_PATH
    faiss_path = config.FAISS_INDEX_PATH
    
    if not os.path.exists(nodes_path) or not os.path.exists(faiss_path):
        print("Error: nodes_store.pkl or faiss_index does not exist!")
        return

    print(f"Loading nodes from {nodes_path}...")
    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)
    print(f"Loaded {len(nodes)} nodes.")

    print(f"Loading FAISS index from {faiss_path}...")
    index = faiss.read_index(faiss_path)
    print(f"FAISS index loaded. Total vectors: {index.ntotal}, dimension: {index.d}")

    if len(nodes) != index.ntotal:
        print(f"Warning: Count mismatch! Nodes ({len(nodes)}) != FAISS vectors ({index.ntotal})")
        # 如果不一致，我們只取較小的那個
        count = min(len(nodes), index.ntotal)
    else:
        count = len(nodes)

    # 重建所有向量
    print("Reconstructing embeddings from FAISS index...")
    embeddings = np.empty((count, index.d), dtype=np.float32)
    for i in range(count):
        embeddings[i] = index.reconstruct(i)

    print("Hashing and caching...")
    cache_keys = []
    for i in range(count):
        node = nodes[i]
        text = node.get_content()
        h = hashlib.sha256(f"RETRIEVAL_DOCUMENT:3072:{text}".encode("utf-8")).hexdigest()
        cache_keys.append(h)

    # 寫入快取檔案
    cache_keys_path = os.path.join(config.INDEX_DIR, "embeddings_cache_keys.pkl")
    cache_values_path = os.path.join(config.INDEX_DIR, "embeddings_cache_values.npy")
    
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    with open(cache_keys_path, "wb") as f:
        pickle.dump(cache_keys, f)
    np.save(cache_values_path, embeddings)
    
    print(f"Successfully saved {len(cache_keys)} keys to {cache_keys_path}")
    print(f"Successfully saved {embeddings.shape} array to {cache_values_path}!")

if __name__ == "__main__":
    init_cache()
