#1、讀取文件內容:

with open("knowledge/文件.txt" , encoding='utf-8' , mode='r') as fp:
    data=fp.read()

#2、根據換行分段

chunk_list = data.split("\n\n")
chunk_list = [chunk for chunk in chunk_list if chunk]
print(chunk_list)