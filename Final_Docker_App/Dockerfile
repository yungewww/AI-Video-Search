## 使用官方 Python 运行时作为父镜像
#FROM python:3.9
## 设置工作目录
#WORKDIR /app
## 将当前目录内容复制到位于 /app 中的容器中
#COPY . /app
#COPY insert_data.py /app/insert_data.py
## 安装 requirements.txt 中指定的任何所需包
#RUN python3 -m pip install --no-cache-dir -r requirements.txt
## 使端口 5000 可用于外部世界
#EXPOSE 5000
## 运行 app.py 时运行应用程序
#CMD ["python", "app.py"]

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装项目依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制当前目录下的所有文件到容器的工作目录
COPY . .

# 对外暴露端口
EXPOSE 8888

# 设置环境变量
ENV FLASK_APP=app.py

# 启动 Flask 应用
CMD ["flask", "run", "--host=0.0.0.0", "--port=8888"]



