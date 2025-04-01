
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import chat_agent_executor
from tool.db.DbTool import DbTool

import logging
from langchain.globals import set_debug

# # 启用LangChain调试模式
# set_debug(True)
#
# # 设置日志级别
# logging.basicConfig(level=logging.DEBUG)

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)

# 初始化 DbTool 并获取连接参数
dbtool = DbTool()
dburl = dbtool.get_url()
db = SQLDatabase.from_uri(dburl)

# 创建工具
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools() # 这个工具执行逻辑的本质也是【通过内置提示词调用大模型】

# 使用agent完成整个数据库的整合
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""


# 创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(
    model,
    tools,
)

# 执行时包含系统消息（关键修改点）
resp = agent_executor.invoke({
    'messages': [
        SystemMessage(content=system_prompt),
        HumanMessage(content='套餐1都有哪些菜品？')
    ]
})

print(resp['messages'])

final_answer = next( # 从生成器中获取第一个符合条件的元素
    (
        msg.content  # 获取msg的content消息内容
        for msg in reversed(resp['messages'])  # 从最后到第一个遍历
        if isinstance(msg, AIMessage)  # 找到第一个 AIMessage 赋值给 msg
    ),
    "未找到有效回答"
)
print("最终答案：", final_answer)

