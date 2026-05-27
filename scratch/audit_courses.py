# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from rag.data_loader import parse_course_file

def audit_fields():
    courses_dir = config.COURSES_DIR
    txt_files = []
    for root, dirs, files in os.walk(courses_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    
    total = len(txt_files)
    print(f"Total files: {total}")
    
    sample_nodes = []
    has_hours = 0
    has_note = 0
    
    for filepath in txt_files:
        try:
            res = parse_course_file(filepath)
            meta = res["metadata"]
            sec = res["sections"]
            
            if meta.get("hours") != "未知":
                has_hours += 1
            if meta.get("note"):
                has_note += 1
                
            if len(sample_nodes) < 5:
                sample_nodes.append((filepath, meta, sec["basic_info"]))
        except Exception as e:
            print(f"Error parsing {os.path.basename(filepath)}: {e}")
            
    print("\n--- Field Extraction Metrics ---")
    print(f"Courses with extracted hours: {has_hours} / {total} ({has_hours/total*100:.2f}%)")
    print(f"Courses with extracted note: {has_note} / {total} ({has_note/total*100:.2f}%)")
    
    print("\n--- Detailed Extraction Samples (First 5) ---")
    for i, (path, meta, basic_info) in enumerate(sample_nodes):
        print(f"\n[Sample {i+1}] File: {os.path.basename(path)}")
        print(f"  - Hours extracted: {meta.get('hours')}")
        print(f"  - Note extracted: {meta.get('note')}")
        print("  - Basic Info Preview:")
        for line in basic_info.splitlines()[:15]:
            print(f"    {line}")

if __name__ == "__main__":
    audit_fields()
