from langchain.tool.db.DbTool import DbTool

str = "SELECT * FROM employee"

dbtool = DbTool()
dbtool.execute_query(str)