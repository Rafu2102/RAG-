# -*- coding: utf-8 -*-
import subprocess
import sys
import time

def run_both():
    print("🚀 正在同一個終端機啟動雙平台機器人 (Discord + Telegram)...")
    
    # 注入跨平台通訊通行證
    import os
    env = os.environ.copy()
    env["ENABLE_BOT_IPC"] = "1"
    env["IPC_PORT"] = "50505"
    
    # 使用當前啟動的 python 環境來啟動腳本
    discord_process = subprocess.Popen([sys.executable, "discord_bot.py"], env=env)
    print(f"✅ Discord Bot 處理程序已啟動 (PID: {discord_process.pid})")
    
    # 稍微等一下避免兩個 bot 同時載入 AI 模型或讀取巨量陣列打架
    time.sleep(2)
    
    telegram_process = subprocess.Popen([sys.executable, "telegram_bot.py"], env=env)
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
