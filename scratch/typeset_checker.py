# -*- coding: utf-8 -*-
import re
import sys

def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    chinese_chars = r'[\u4e00-\u9fa5]'
    english_chars = r'[a-zA-Z0-9]'
    
    violations = []
    
    for idx, line in enumerate(lines, 1):
        line_clean = line.strip()
        if not line_clean:
            continue
        # Skip markdown code blocks, links, inline code and image blocks
        if line_clean.startswith("```") or line_clean.startswith("import ") or line_clean.startswith("url =") or line_clean.startswith("├──") or line_clean.startswith("└──") or line_clean.startswith("│"):
            continue
            
        # 1. Chinese + English/Number
        for m in re.finditer(f'({chinese_chars})({english_chars})', line):
            # Exclude specific patterns if needed, but let's report them
            violations.append(f"Line {idx}: Missing space between Chinese and English/Number: '{line[m.start()-2:m.end()+2].strip()}'")
            
        # 2. English/Number + Chinese
        for m in re.finditer(f'({english_chars})({chinese_chars})', line):
            # Except % and °
            char = line[m.start():m.start()+1]
            next_char = line[m.start()+1:m.start()+2]
            if char in ['%', '°']:
                continue
            # Also exclude if it's within a markdown link e.g. [text](url) or inside `code` or HTML tags
            # Simple heuristic: check if it's a bracket or URL component
            violations.append(f"Line {idx}: Missing space between English/Number and Chinese: '{line[max(0, m.start()-2):m.end()+2].strip()}'")
            
        # 3. Check number + unit space (like 225KB -> 225 KB)
        for m in re.finditer(r'(\d+)([\u4e00-\u9fa5a-zA-Z]+)', line):
            num = m.group(1)
            unit = m.group(2)
            # Allow % or if the unit is part of a variable or path or URL or hex (like RTX 4060, v1beta, 2026-04-13)
            # Filter out common false positives
            if unit.startswith('%') or unit.startswith('°'):
                continue
            if re.match(r'^[a-fA-F0-9]+$', num + unit): # hex
                continue
            if any(x in line for x in ['http', 'venv', 'credentials', 'groups', 'manifest', '3.5', 'Port 50505']):
                continue
            if unit.lower() in ['g', 'gb', 'mb', 'kb', 'ms', 's', 'rpm', 'tpm', 'vram']:
                violations.append(f"Line {idx}: Missing space between number and unit: '{m.group(0)}' (should be '{num} {unit}')")
                
    return violations

if __name__ == "__main__":
    filepath = r"d:\AI HYBRID\README.md"
    violations = check_file(filepath)
    if violations:
        print(f"Found {len(violations)} potential typesetting violations in {filepath}:")
        for v in violations[:30]:
            print(" -", v)
        if len(violations) > 30:
            print(f" ... and {len(violations) - 30} more.")
    else:
        print("No typesetting violations found!")
