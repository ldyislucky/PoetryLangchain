import os

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# 聊天机器人案例
# 创建模型

# 1、加载数据: 一篇公众号内容数据
loader = WebBaseLoader(
    web_paths=['https://mp.weixin.qq.com/s?__biz=MzA4ODY4MDE0NA%3D%3D&mid=2247640627&idx=3&sn=fbcdbbe55769ce335dc8f7d99b576e25&chksm=91501849f1697e2015bbf648ef2d77d44ff2cc71332d002f24de18e300c6cb45cec54496a2cf&scene=27'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(id=('activity-name', 'js_content'))
    )
)

docs = loader.load()

print(len(docs))
print(docs)

# 2、大文本的切割
# text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

for split in splits:
    print(split)


from sentence_transformers import SentenceTransformer

# 定义查询和文档集合
queries = ['query_1', 'query_2']  # 查询列表，每个元素为一个查询字符串
passages = ["样例文档-1", "样例文档-2"]  # 文档列表，每个元素为一个文档字符串
instruction = "为这个句子生成表示以用于检索相关文章："  # 指令字符串，用于增强查询的语义信息

# 加载本地预训练模型
model_path = "D:/D/document/donotdelete/models/bge-small-zh/bge-small-zh-v1.5"

"""
模型路径：指向本地存储的SentenceTransformer模型文件夹，需包含pytorch_model.bin等必要文件；
如果本地没有模型，根据官网示例直接写模型名称就行，但是这样老报模型匹配不上的错
所以还是用本地方式就行，不要用官网的示例
"""
model = SentenceTransformer(model_path)

# 对查询进行编码，生成查询嵌入向量
# 将指令与每个查询拼接后编码，并对嵌入向量进行归一化处理
q_embeddings = model.encode([instruction + q for q in queries], normalize_embeddings=True)

# 对文档集合进行编码，生成文档嵌入向量
# 对文档列表进行编码，并对嵌入向量进行归一化处理
p_embeddings = model.encode(passages, normalize_embeddings=True)

# 计算查询与文档之间的相似度分数
# 使用矩阵乘法计算查询嵌入与文档嵌入的余弦相似度
scores = q_embeddings @ p_embeddings.T

# 输出每个查询与文档的相似度分数
for query, score in zip(queries, scores):
    print(f"query: {query}")  # 打印当前查询
    for passage, score in zip(passages, score):
        # 打印每个文档及其与当前查询的相似度分数
        print(f"passage: {passage}, score: {score}")


