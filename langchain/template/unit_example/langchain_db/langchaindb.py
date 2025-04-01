
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableLambda

from tool.db.DbTool import DbTool

import logging
from langchain.globals import set_debug

# 启用LangChain调试模式
set_debug(True)

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)




# 初始化模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)

# 初始化 DbTool 并获取连接参数
dbtool = DbTool()
dburl = dbtool.get_url()

db = SQLDatabase.from_uri(dburl)
# 创建 SQL 查询链
chain = create_sql_query_chain(model, db)#.with_config({"verbose": True})
question = "请查询所有菜品?"

# result = chain.invoke({"question": "请查询所有菜品"})
# print(result)
result = chain.invoke({"question": question})
result = result.replace("SQLQuery: ", "")
print(result)

answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)
# 创建一个执行sql语句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)

# 1、生成SQL，2、执行SQL
# 2、模板


# 将 result 包装为 RunnableLambda
chain1 = (
    RunnablePassthrough.assign(query=RunnableLambda(lambda _: result))   # 此处添加了 query 键值对
    .assign(result=lambda inputs: execute_sql_tool.invoke(inputs["query"]))   # 此处添加了 result 键值对，传入的参数为query键值对的值
    | answer_prompt
    | model
    | StrOutputParser()
)#.with_config({"verbose": True})



rep = chain1.invoke(input={'question': question})
print(rep)
