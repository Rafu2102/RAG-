# -*- coding: utf-8 -*-
import sys
import os
import re
import html

# 模擬修復後的函數
def discord_md_to_tg_html_fixed(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r'```(?:\w+\n)?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'^#+\s+(.*)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    
    # 處理 Markdown 連結，過濾 URL 中的 &lt;、&gt;、<、> 符號
    def replace_link(match):
        title = match.group(1)
        url = match.group(2)
        url = url.replace("&lt;", "").replace("&gt;", "").replace("<", "").replace(">", "").strip()
        return f'<a href="{url}">{title}</a>'
        
    text = re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)
    
    # 處理被 &lt; 和 &gt; 包裹的裸露 URL
    text = re.sub(
        r'(?<!href=")&lt;(https?://[^\s&<>]+)&gt;',
        r'<a href="\1">\1</a>',
        text
    )
    return text

def test_conversion():
    # 測試1：一般 Markdown 連結
    md1 = "[金大官網](https://www.nqu.edu.tw)"
    html1 = discord_md_to_tg_html_fixed(md1)
    print(f"Test 1:\nInput:  {md1}\nOutput: {html1}\n")

    # 測試2：包裝了 <> 的連結 (通常由 _suppress_discord_embeds 產生)
    md2 = "[金大公告](<https://www.nqu.edu.tw/p/123>)"
    html2 = discord_md_to_tg_html_fixed(md2)
    print(f"Test 2:\nInput:  {md2}\nOutput: {html2}\n")

    # 測試3：裸露的 URL 被 <> 包裝
    md3 = "<https://www.dcard.tw/f/nqu/p/123456>"
    html3 = discord_md_to_tg_html_fixed(md3)
    print(f"Test 3:\nInput:  {md3}\nOutput: {html3}\n")

if __name__ == "__main__":
    test_conversion()
