
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

# 聊天机器人案例
# 创建模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)

# 1、加载数据: 一篇公众号内容数据
loader = WebBaseLoader(
    web_paths=['https://mp.weixin.qq.com/s?__biz=MzA4ODY4MDE0NA%3D%3D&mid=2247640627&idx=3&sn=fbcdbbe55769ce335dc8f7d99b576e25&chksm=91501849f1697e2015bbf648ef2d77d44ff2cc71332d002f24de18e300c6cb45cec54496a2cf&scene=27'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(id=('activity-name ', 'js_content'))
    )
)

docs = loader.load()

# print(len(docs))
print(docs)

# 2、大文本的切割
# text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 2、指定使用的预训练模型名称
model_name = "D:/D/document/donotdelete/models/bge-small-zh/bge-small-zh-v1.5"

# 配置模型加载参数，指定设备为 GPU（如果可用），则需修改为 'cuda'
model_kwargs = {'device': 'cpu'}

# 配置编码参数，设置 normalize_embeddings 为 True 以支持余弦相似度计算
encode_kwargs = {'normalize_embeddings': True}

# 初始化嵌入语义理解模型（使用镜像或本地路径），传入模型名称、加载参数、编码参数和查询指令
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# 3、存储为document向量库
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
# 打印存储库信息
print(vectorstore.get())

# 4、创建向量库检索器
retriever = vectorstore.as_retriever()

# 整合链
# 创建一个聊天提示模板，用于生成聊天消息
system_prompt = """你是一个用于问答任务的助手。
使用以下检索到的上下文回答问题。如果你不知道答案，就说你不知道。
最多用三个句子，保持答案简洁。.\n

{context}
"""
prompt = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),  #
        ("human", "{input}"),
    ]
)

# 得到chain
chain1 = create_stuff_documents_chain(model, prompt)

# retriever 根据输入从向量库中检索相关的文档片段（context）送给chain1，然后送给chain1结合（context）、用户问题、prompt进行回答
# chain2 = create_retrieval_chain(retriever, chain1)
#
# # resp = chain2.invoke({'input': "什么是任务分解?"})
# resp = chain2.invoke({'input': "什么是深度神经网络?"})
#
# print(resp['answer'])


"""
查询检索器也需要对话上下文才能被理解。
假设我们有一个多轮对话场景：
第一轮对话 用户提问： 什么是任务分解？ 系统回答： 任务分解是将一个复杂任务拆分为多个简单子任务的过程。
第二轮对话 用户接着提问： 它有哪些应用场景？
在这个例子中，用户的问题“它有哪些应用场景？”中的“它”指代的是上一轮对话中提到的“任务分解”。如果检索器没有对话上下文，它无法知道“它”具体指的是什么，可能会返回无关的结果。
以下是解决方案：
"""


