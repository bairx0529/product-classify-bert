import uvicorn
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.predict_router import predict_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

# 1) API 路由
app.include_router(predict_router)

# 2) 静态资源 / 模板
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def run_app():
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
