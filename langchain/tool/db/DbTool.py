
import mysql.connector
from mysql.connector import Error

class DbTool:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='reggie'
        )
        if self.conn.is_connected():
            self.cursor = self.conn.cursor()
            print("数据库连接成功")
        else:
            print("数据库连接失败")

    def execute_query(self, query: str):
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            return result
        except Error as e:
            print("Error:", e)
            return None
    def close(self):
        if self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            print("数据库连接已关闭")
        else:
            print("数据库连接已关闭")