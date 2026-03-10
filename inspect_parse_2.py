import os
from data_loader import parse_course_file

filepath = r"d:\AI HYBRID\data\資工系113學年度第2學期課程資訊\人工智慧實務 (Practical Artificial Intelligence).txt"
try:
    data = parse_course_file(filepath)
    print("--- GRADING ---")
    print(repr(data["sections"]["grading"]))
except Exception as e:
    print("ERROR:", e)
