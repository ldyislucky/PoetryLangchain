from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_deepseek import ChatDeepSeek

from tool.db.DbTool import DbTool

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)

# 初始化 DbTool 并获取连接参数
dbtool = DbTool()
dburl = dbtool.get_url()

db = SQLDatabase.from_uri(dburl)
# 创建 SQL 查询链
chain = create_sql_query_chain(model, db)

# 执行查询
# 执行查询
result = chain.invoke({"question": "请查询所有菜品"})
print(result)
result = chain.invoke({"question": "请问：员工表中有多少条数据？"})
print(result)


