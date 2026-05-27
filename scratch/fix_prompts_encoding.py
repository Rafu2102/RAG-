# -*- coding: utf-8 -*-
import os

def fix_prompts():
    file_path = os.path.join("llm", "prompts.py")
    print(f"Reading {file_path}...")

    # 嘗試用 CP950 (Big5) 讀取，因為在 Windows 環境下最容易被寫成 CP950
    try:
        with open(file_path, "r", encoding="cp950") as f:
            content = f.read()
        print("Successfully read with cp950!")
        
        # 預覽前10行和第13行附近
        lines = content.splitlines()
        print("\n=== First 15 lines read as cp950 ===")
        for i, line in enumerate(lines[:15]):
            print(f"{i+1}: {line}")
            
        # 檢查是否有 unicode error 位元組被寫回
        # 我們將其以 utf-8 寫回
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("\nSuccessfully rewrote llm/prompts.py as UTF-8!")
        return True
    except Exception as e:
        print(f"Failed to read with cp950: {e}")

    # 如果 CP950 失敗，試試用 utf-8 (ignore errors) 來讀，然後再覆寫
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print("Read with utf-8 (ignore errors)")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Rewrote as UTF-8 (cleaned up bad bytes)")
        return True
    except Exception as e:
        print(f"Failed utf-8 backup: {e}")
        return False

if __name__ == "__main__":
    fix_prompts()
