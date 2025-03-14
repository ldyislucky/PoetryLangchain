from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_deepseek import ChatDeepSeek
from langserve import add_routes

app = FastAPI(
    title="李东洋的服务器",
    description="使用了Langchain的简单服务器",
    version="1.0"
)

add_routes(
    app,
    ChatDeepSeek(model="deepseek-chat",max_tokens=100),
    path="/deepseekapi",
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
