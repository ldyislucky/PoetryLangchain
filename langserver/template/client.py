import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

promt = ChatPromptTemplate.from_template("讲一个关于{type}笑话,50字以内。以json格式返回")
#langserver 的url调用方式
model = RemoteRunnable("http://127.0.0.1:8000//deepseekapi")
parser = JsonOutputParser()

# chain = promt |model | JsonOutputParser()  也行
chain = promt |model | parser
async def async_stream():
    async for chunk in chain.astream({"type": "猫"}):
        print(chunk,end="",flush=True)


asyncio.run(async_stream())