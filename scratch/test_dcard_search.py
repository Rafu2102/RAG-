# -*- coding: utf-8 -*-
import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.setup_logging()

from tools.dcard_search_tool import search_dcard_professor

async def main():
    print("=" * 60)
    print("Calling search_dcard_professor('英文教授') directly...")
    result_eng = await search_dcard_professor("英文教授")
    print("\n[Final Result (英文教授)]:")
    print(result_eng)
    print("=" * 60)
    print("Calling search_dcard_professor('李錫捷') directly...")
    result_lee = await search_dcard_professor("李錫捷")
    print("\n[Final Result (李錫捷)]:")
    print(result_lee)
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
