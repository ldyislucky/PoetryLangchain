from langchain_huggingface import HuggingFaceEmbeddings

# 初始化 HuggingFaceBgeEmbeddings 模型的配置和实例化
# 参数说明：
# - model_name: 指定使用的预训练模型名称或路径，例如 "BAAI/bge-large-en-v1.5"。
# - model_kwargs: 模型加载时的额外参数，例如指定设备为 'cuda' 以使用 GPU 加速。
# - encode_kwargs: 编码时的额外参数，例如设置 'normalize_embeddings' 为 True 以计算余弦相似度。
# - query_instruction: 查询指令，用于生成句子表示以支持检索任务。
# 返回值：无（代码片段未包含返回值逻辑）

model_name = "D:/D/document/donotdelete/models/bge-small-zh/bge-small-zh-v1.5"  # 指定使用的预训练模型名称

# 配置模型加载参数，指定设备为 GPU（如果可用），则需修改为 'cuda'
model_kwargs = {'device': 'cpu'}

# 配置编码参数，设置 normalize_embeddings 为 True 以支持余弦相似度计算
encode_kwargs = {'normalize_embeddings': True}

# 实例化 HuggingFaceBgeEmbeddings 模型，传入模型名称、加载参数、编码参数和查询指令
model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

def generate_query_embedding(query: str):
    # 手动拼接查询指令
    full_query = f"为这个句子生成表示以用于检索相关文章：{query}"
    return model.embed_query(full_query)

# 示例调用
query = "这是一段测试文本"
embedding = generate_query_embedding(query)
print(embedding)


