from fastapi import FastAPI, HTTPException
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import chat_agent_executor
from langchain_deepseek import ChatDeepSeek




class DbTool:
    def __init__(self):
        self.host = '127.0.0.1'
        self.user = 'root'
        self.password = 'root'
        self.database = 'reggie'
        self.port = 3306
        self.conn = None
        self.cursor = None

    def get_url(self):
        return f'mysql+mysqldb://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?charset=utf8mb4'





# 初始化 FastAPI 应用
app = FastAPI()

# 初始化模型和数据库工具
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)
dbtool = DbTool()
dburl = dbtool.get_url()
db = SQLDatabase.from_uri(dburl)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 系统提示词
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

@app.post("/query")
async def query_database(question: str):
    """
    接收用户问题，调用 LangChain 代理执行 SQL 查询，并返回结果。
    """
    try:
        # 调用代理处理问题
        resp = agent_executor.invoke({
            'messages': [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
        })

        # 提取最终答案
        final_answer = next(
            (
                msg.content
                for msg in reversed(resp['messages'])
                if isinstance(msg, AIMessage)
            ),
            "未找到有效回答"
        )

        return {"question": question, "answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# 启动命令：uvicorn AgentDb:app --reload
