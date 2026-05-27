import pickle
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

nodes_store_path = r"d:\AI HYBRID\index_store\nodes_store.pkl"

with open(nodes_store_path, "rb") as f:
    nodes = pickle.load(f)

courses_114_2 = []
seen = set()

for node in nodes:
    meta = getattr(node, "metadata", {})
    dept_short = meta.get("dept_short", "")
    academic_year = meta.get("academic_year", "")
    semester = meta.get("semester", "")
    course_name = meta.get("course_name", "")
    
    if academic_year == "114" and semester == "2" and dept_short == "通識中心":
        key = (course_name, meta.get("teacher", ""))
        if key not in seen:
            seen.add(key)
            courses_114_2.append(meta)

print(f"Total unique Tongshi courses in 114-2: {len(courses_114_2)}")
for idx, c in enumerate(courses_114_2[:50]):
    print(f"{idx+1}. {c.get('course_name')} | Teacher: {c.get('teacher')} | Required: {c.get('required_or_elective')} | Class: {c.get('grade_class')}")
