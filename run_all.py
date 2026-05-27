# -*- coding: utf-8 -*-
import subprocess
import sys
import time

def kill_zombie_bots():
    """主動、精確強力掃除背景殘留的舊有 Bot 處理程序與通訊埠佔用"""
    print("🧹 正在掃除背景殘留的舊有 Bot 處理程序...")
    
    # 1. 結束所有命令列含有 discord_bot.py 或 telegram_bot.py 的 Python 進程
    cmd_process = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process -Filter \"CommandLine like '%discord_bot.py%' or CommandLine like '%telegram_bot.py%'\" | Foreach-Object { Stop-Process -Id $_.ProcessId -Force }"
    ]
    
    # 2. 結束佔用 50505 監控 IPC 通訊埠的任何處理程序 (防止 Errno 10048)
    cmd_port = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-NetTCPConnection -LocalPort 50505 -ErrorAction SilentlyContinue | Foreach-Object { Stop-Process -Id $_.OwningProcess -Force }"
    ]
    
    try:
        subprocess.run(cmd_process, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd_port, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ 掃除完成，50505 連接埠與 RAG API 鎖已完全釋放！\n" + "-"*50)
    except Exception as e:
        print(f"⚠️ 掃除處理程序時發生微小異常 (可忽略): {e}\n" + "-"*50)

def run_both():
    # 啟動前進行大掃除
    kill_zombie_bots()
    
    print("🚀 正在同一個終端機啟動雙平台機器人 (Discord + Telegram)...")
    
    # 載入 .env 變數以傳遞給子進程
    import os
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(project_root, ".env"))
    
    # 注入跨平台通訊通行證
    env = os.environ.copy()
    env["ENABLE_BOT_IPC"] = "1"
    env["IPC_PORT"] = "50505"
    
    # 使用當前啟動的 python 環境來啟動腳本，並顯式指定工作目錄
    discord_process = subprocess.Popen([sys.executable, "discord_bot.py"], env=env, cwd=project_root)
    print(f"✅ Discord Bot 處理程序已啟動 (PID: {discord_process.pid})")
    
    # 稍微等一下避免兩個 bot 同時載入 AI 模型或讀取巨量陣列打架
    time.sleep(2)
    
    telegram_process = subprocess.Popen([sys.executable, "telegram_bot.py"], env=env, cwd=project_root)
    print(f"✅ Telegram Bot 處理程序已啟動 (PID: {telegram_process.pid})")
    
    print("\n💡 提示：按 [Ctrl + C] 可以一次把兩個機器人關閉。\n" + "-"*50)
    
    try:
        # 讓這隻母程式停在這裡等，如果 bot 當機了會在這裡捕獲
        discord_process.wait()
        telegram_process.wait()
    except KeyboardInterrupt:
        # 攔截你在終端機按 Ctrl+C，優雅地關閉兩個小弟
        print("\n🛑 收到終止訊號 (Ctrl+C)，準備關閉兩個機器人...")
        discord_process.terminate()
        telegram_process.terminate()
        discord_process.wait()
        telegram_process.wait()
        print("✅ 關閉完成")

if __name__ == "__main__":
    run_both()

