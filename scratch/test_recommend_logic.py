# -*- coding: utf-8 -*-
import sys
import os
import asyncio

# 將專案根目錄加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm.coreference import resolve_coreference
from rag.query_router import route_and_rewrite
from rag.retriever import hybrid_retrieve
from rag.index_manager import load_and_index


async def test_coreference():
    print("\n--- 🧪 測試 1: 指代消解 (Coreference Resolution) ---")
    chat_history = [
        {"role": "user", "content": "我想修資工系的 Linux 課。"},
        {"role": "assistant", "content": "資工系在 114 學年度有開設 Linux 系統管理與實務。"}
    ]
    
    # 測試明確的科系切換提問
    question_1 = "長照系推薦課程"
    res_1 = await resolve_coreference(question_1, chat_history)
    print(f"原始問題：「{question_1}」")
    print(f"消解結果：「{res_1}」")
    assert "資工" not in res_1, "❌ 錯誤：指代消解將歷史對話中的資工系混入了長照系提問！"
    print("✅ 測試 1 通過！沒有發生跨科系污染。")


async def test_query_router():
    print("\n--- 🧪 測試 2: Query Router 路由與安全閥 ---")
    # 測試指定科系推薦
    q_1 = "長照系推薦課程"
    result_1, queries_1 = await route_and_rewrite(q_1, user_profile={"department": "資工系", "grade": "2"})
    print(f"提問：「{q_1}」")
    print(f"路由屬性：query_type={result_1.query_type}, dept_short={result_1.metadata_filters.get('dept_short')}, is_career_planning={result_1.is_career_planning}")
    
    assert result_1.is_career_planning is False, "❌ 錯誤：指定科系推薦卻誤觸了 is_career_planning = True！"
    assert result_1.metadata_filters.get("dept_short") == "長照系", f"❌ 錯誤：科系過濾條件丟失或不正確！得到：{result_1.metadata_filters.get('dept_short')}"
    print("✅ 測試 2.1 (指定科系推薦) 通過！")

    # 測試跨域職涯探索
    q_2 = "我想學人工智慧，有推薦修什麼課嗎？"
    result_2, queries_2 = await route_and_rewrite(q_2, user_profile={"department": "企管系", "grade": "2"})
    print(f"提問：「{q_2}」")
    print(f"路由屬性：query_type={result_2.query_type}, dept_short={result_2.metadata_filters.get('dept_short')}, is_career_planning={result_2.is_career_planning}")
    
    assert result_2.is_career_planning is True, "❌ 錯誤：泛化職涯探索應將 is_career_planning 判定為 True！"
    print("✅ 測試 2.2 (泛化職涯探索) 通過！")


async def test_retriever():
    print("\n--- 🧪 測試 3: Retriever 抽樣限制 (Full Curriculum Scan) ---")
    print("📂 正在載入索引資料...")
    nodes, faiss_index, bm25_index = load_and_index()
    from rag.query_router import init_known_registry
    init_known_registry(nodes)
    
    # 模擬職涯探索
    q = "我想當 AI 工程師"
    route_result, queries = await route_and_rewrite(q, user_profile={"department": "資工系", "grade": "2"})
    
    print("🔍 執行 Hybrid Retrieve (職涯探索)...")
    chunks = hybrid_retrieve(
        queries=queries,
        route_result=route_result,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        nodes=nodes,
        user_profile={"department": "資工系", "grade": "2"}
    )
    
    # 統計抽取的各科系非本科系課程數
    other_depts_courses = {}
    
    def is_primary(d):
        d_clean = d.replace("系所", "").replace("學系", "").replace("系", "").replace("資訊工程", "資工").strip()
        return "資工" in d_clean

    for chunk in chunks:
        meta = chunk.node.metadata
        dept = meta.get("dept_short", meta.get("department", "未知"))
        cname = meta.get("course_name")
        if not is_primary(dept) and cname:
            if dept not in other_depts_courses:
                other_depts_courses[dept] = set()
            other_depts_courses[dept].add(cname)
            
    total_other_courses = sum(len(courses) for courses in other_depts_courses.values())
    print(f"非本科系 (資工系/資工碩除外) 抽樣抽取課程分佈：")
    for d, courses in other_depts_courses.items():
        print(f"  - {d}: {len(courses)} 門課 -> {list(courses)}")
    print(f"非本科系抽樣課程總數：{total_other_courses} 門課")
    
    assert total_other_courses <= 15, f"❌ 錯誤：非本科系抽取課程數為 {total_other_courses}，超出了 15 門課的限制上限！"
    print("✅ 測試 3 通過！非本科系課程限制在 15 門課以內，防範了 Context 過載。")


async def main():
    config.setup_logging()
    try:
        await test_coreference()
        await test_query_router()
        await test_retriever()
        print("\n✨✨ 所有測試全部成功通過！ ✨✨")
    except AssertionError as e:
        print(f"\n❌ 測試失敗：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 執行發生未預期錯誤：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
