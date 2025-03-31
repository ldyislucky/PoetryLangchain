
import mysql.connector
from mysql.connector import Error

try:
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='reggie'
    )
    # 连接成功时
    if conn.is_connected():
        # 创建游标（应该相当于打开了一个会话）对象用于执行SQL语句
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employee")
        print(cursor.fetchall())
except Error as e:
    print("Error:", e)
finally:
    # 资源清理：无论是否发生异常都执行连接关闭操作
    if conn.is_connected():
        cursor.close()
        conn.close()