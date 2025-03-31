from tool.db.DbTool import DbTool

str = "SELECT * FROM employee"

dbtool = DbTool()
list = dbtool.execute_query(str)
for row in list:
    print(row)

dbtool.close()