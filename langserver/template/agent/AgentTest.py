"""
存在弃用风险，暂时无法解决
"""

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import initialize_agent, AgentType

# 定义提示模板（无需修改，但需确保代理正确使用）
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个乐于助人的助手。用中文尽你所能简洁回答问题。注意：
                1. 如果问题无需外部信息就直接给出答案，
                2. 如果问题需要外部信息，使用工具调用。
                3. 当获得足够信息后，直接给出最终答案，停止工具调用。
                """),
    HumanMessage(content="{input}"),  # 注意这里的变量名是 {input}
])

# 创建模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)

# 创建搜索工具
search = TavilySearchResults(max_results=2)
tools = [
    Tool(
        name="TavilySearch",
        description="搜索网络获取最新信息",
        func=search.run
    )
]

# 创建代理（使用 LangChain 原生方法）
agent_executor = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5  # 递归限制
)

# 调用代理时，输入格式改为 {'input': '问题内容'}
resp = agent_executor.invoke({'input': '中国最新的首都是哪个城市？'})
print("首都是：", resp['output'])

resp2 = agent_executor.invoke({'input': '北京天气怎么样？'})
print("北京天气：", resp2['output'])

resp2 = agent_executor.invoke({'input': '我的第一个问题是什么？'})
print("上文信息：", resp2['output'])