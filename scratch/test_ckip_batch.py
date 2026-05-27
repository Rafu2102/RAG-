# -*- coding: utf-8 -*-
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from nlp_utils import tokenize_texts_ckip

def test():
    config.setup_logging()
    print("Loading model...")
    # 測試幾筆數據
    texts = ["金門大學資訊工程學系柯志亨老師開設的互動式網頁程式設計課程時間是星期三的第五節到第七節。"] * 100
    
    start = time.time()
    res = tokenize_texts_ckip(texts)
    duration = time.time() - start
    print(f"Tokenized {len(texts)} texts in {duration:.2f} seconds.")
    print(f"Speed: {len(texts)/duration:.2f} texts/sec")

if __name__ == "__main__":
    test()
