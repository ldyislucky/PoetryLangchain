from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage, HumanMessage

# 初始化模型
llm = ChatDeepSeek(model="deepseek-chat", max_tokens=500)

# 模拟导演信息数据库
director_db = {
    "诺兰": {"birth_year": 1970, "movies": ["奥本海默", "星际穿越"]},
    "克里斯托弗·诺兰": {"birth_year": 1970, "movies": ["奥本海默", "星际穿越"]},
    "Christopher Nolan": {"birth_year": 1970, "movies": ["奥本海默", "星际穿越"]},
    # 可扩展其他导演信息
}

def search_director_info(query: str):
    """查找导演信息的工具（带模拟数据）"""
    info = director_db.get(query)
    if info:
        return {"status": "success", "data": info}
    else:
        return {"status": "failure", "message": "未找到该导演信息"}

# 创建工具列表
tools = [
    Tool(
        name="SearchDirectorInfo",
        func=search_director_info,
        description="查找导演相关信息的工具"
    )
]

# 构建提示模板
prompt = ChatPromptTemplate.from_messages([
    # 系统指令
    SystemMessage(content="""
    你是一位电影专家，擅长查找导演信息。请按以下步骤操作：
    1. 使用工具 `SearchDirectorInfo` 查找导演的出生年份。
    2. 如果工具返回的是字典，提取其中的 `birth_year` 键值。
    3. 如果返回结果的"status"为"success"，用当前年份 2023 减去出生年份计算年龄；否则回复未查询到导演信息
    4. 用中文简洁回答问题。
    """),
    # 记录用户与代理之前的对话，保持上下文连贯。插入对话历史
    MessagesPlaceholder(variable_name="chat_history"),
    # 当前用户输入
    HumanMessage(content="{input}"),
    # 插入代理的中间思考
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建代理
agent = create_tool_calling_agent(
    llm,
    tools=tools,  # 添加 tools 参数
    prompt=prompt
)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,  # 添加 tools 参数，两个 tools 参数必须一致，不能省略
    verbose=True,  # 设置详细模式,查看代理的每一步推理过程
    max_iterations=5  # 设置最大迭代次数
)

# 执行查询
try:
    response = agent_executor.invoke({
        "input": "谁执导了2023年的电影《奥本海默》，他多少岁了？",
        "chat_history": []  # 添加空列表初始化对话历史
    })
    print("答案:", response["output"])
except Exception as e:
    print("错误:", str(e))

