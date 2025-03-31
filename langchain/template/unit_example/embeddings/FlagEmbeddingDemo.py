from FlagEmbedding import FlagModel

# 初始化FlagModel模型
# 参数说明：
# - model_path: 模型路径，指向预训练模型的存储位置
# - query_instruction_for_retrieval: 为检索任务添加的指令，用于生成查询的表示，对于中文来说，这个提示词并没有啥影响，但是英文就有影响了
# - use_fp16: 是否使用半精度浮点数（FP16）加速计算，可能会略微降低性能
model = FlagModel('D:/D/document/donotdelete/models/bge-large-zh/bge-large-zh-v1.5',
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)

# 定义两个句子列表，用于生成嵌入向量
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-1", "样例数据-4"]

# 使用模型对句子列表进行编码，生成对应的嵌入向量
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)

# 计算两个嵌入向量集合的相似度矩阵
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

print("\n===============================\n")

# 对于短查询到长文档（s2p）的检索任务，建议使用encode_queries()方法
# 该方法会自动为每个查询添加指令，适合处理短查询
# 而文档集合可以继续使用encode()或encode_corpus()方法，因为它们不需要额外指令
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]

# 对查询和文档分别生成嵌入向量
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)

# 计算查询与文档之间的相似度得分矩阵
scores = q_embeddings @ p_embeddings.T
print(scores)
