
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
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
model_name = "D:/D/document/donotdelete/models/bge-large-zh/bge-large-zh-v1.5"

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
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),    # 占位符，用于插入聊天历史记录
        ("human", "{input}"),
    ]
)

# 2. 创建文档处理链（chain1）
chain1 = create_stuff_documents_chain(
    model,          # 模型：基于检索到的文档和 ‌原始用户问题‌ 生成最终回答
    prompt          # 提示模板：结合context和历史聊天记录生成回答
)

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


# 创建一个子链
# 子链的提示模板
contextualize_q_system_prompt = """请根据已有的聊天记录和最新的用户
问题（其中可能涉及对聊天历史上下文的引用），将其重新表述为一个无需
依赖聊天记录即可独立理解的问題。仅需重新调整表述（若有必要），否则
直接返回原问题，请勿回答问题本身。
"""

retriever_history_Prompt = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),  # 占位符，用于插入聊天历史记录
        ("human", "{input}"),
    ]
)

# 1. 创建历史感知检索子链
history_chain = create_history_aware_retriever(
    model,          # 模型：用于重写问题，重写后的问题用于检索文档，不会传递给chain1
    retriever,      # 检索器：从向量库获取文档
    retriever_history_Prompt  # 提示模板：指导如何结合历史
)

# 保存问答的历史记录
store = {}
config={'configurable': {'session_id': 'ls123456'}}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 3. 整合成完整检索链（chain）
chain = create_retrieval_chain(
    history_chain,  # 先执行：处理历史 + 检索相关的文档片段（context）送给chain1
    chain1          # 后执行：生成回答
)

# RunnableWithMessageHistory用于管理带有聊天历史记录的对话链（chain）。它的主要功能是将聊天历史记录整合到对话流程中
result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',  # 获取用户输入的问题，存放到("human", "{input}"),中
    history_messages_key='chat_history',  # 指定聊天历史记录的键名，存放get_session_history返回的结果，然后放到MessagesPlaceholder('chat_history')中
    output_messages_key='answer'
)

# 第一轮对话
resp1 = result_chain.invoke(
    {'input': '什么是深度神经网络?'},
    config=config
)

print(resp1['answer'])

# 第二轮对话
resp2 = result_chain.invoke(
    {'input': '它有什么作用?'},
    config=config
)

print(resp2['answer'])


# 第三轮对话
resp3 = result_chain.invoke(
    {'input': '他将来会怎样影响我们的生活?'},
    config=config
)

print(resp3['answer'])

print("\n\n==================================\n\n")
print(store)