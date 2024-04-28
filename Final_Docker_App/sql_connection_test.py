import psycopg2
import numpy as np

# 数据库连接参数，根据您的环境进行调整
dbname = 'mydatabase'  # 数据库名称
user = 'myuser'  # 数据库用户名
password = 'mypassword'  # 如果有密码的话
host = 'localhost'  # 或者 Docker 容器的 IP 地址
port = '5432'  # PostgreSQL 默认端口

# 建立数据库连接
conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

# 创建 cursor 以执行查询
cur = conn.cursor()

# 编写 SQL 查询以获取 image_vector 表中最后一行的 vector 字段
sql_query = "SELECT vector FROM image_vector"

# 执行查询
cur.execute(sql_query)

# 获取查询结果
result = cur.fetchone()

if result:
    # 提取 vector 字段
    vector_string = result[0]

    # 转换字符串到 NumPy 数组
    # 假设 vector 字段存储的形式是 '1.0,2.0,3.0' 这样的形式
    image_vector = np.fromstring(vector_string, sep=',')

    print(image_vector)  # 输出 NumPy 数组

# 关闭 cursor 和连接
cur.close()
conn.close()
