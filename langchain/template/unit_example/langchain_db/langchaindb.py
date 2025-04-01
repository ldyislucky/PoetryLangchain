
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableLambda

from tool.db.DbTool import DbTool

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)

# 初始化 DbTool 并获取连接参数
dbtool = DbTool()
dburl = dbtool.get_url()

db = SQLDatabase.from_uri(dburl)
# 创建 SQL 查询链
chain = create_sql_query_chain(model, db)
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
chain = (
    RunnablePassthrough.assign(query=RunnableLambda(lambda _: result))
    .assign(result=lambda inputs: execute_sql_tool.invoke(inputs["query"]))
    | answer_prompt
    | model
    | StrOutputParser()
)



rep = chain.invoke(input={'question': question})
print(rep)
