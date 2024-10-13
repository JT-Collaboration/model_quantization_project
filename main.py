from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import os
from pathlib import Path
from quantization import quantize_model

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)  # 创建文件夹用于存储上传的文件

model_name = ''


# 上传模型
@app.post("/upload/")
async def upload_model(file: UploadFile = File(...)):
    try:
        # 检查文件扩展名
        if not file.filename.endswith(".pt"):
            raise HTTPException(status_code=400, detail="Only .pt files are allowed")

        # 将上传的文件保存到服务器
        global model_name  # 全局变量，保存模型名字
        model_name = os.path.splitext(file.filename)[0]
        # print(model_name)
        model_path = UPLOAD_FOLDER / file.filename
        with open(model_path, "wb") as buffer:
            buffer.write(await file.read())

        # 对模型进行量化
        quantized_model_path = quantize_model(model_path, model_name)

        return {"quantized_model_path": str(quantized_model_path)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")


# 下载量化后的模型
@app.get("/download/")
async def download_model(quantized_model: str):
    if not os.path.exists(quantized_model):
        raise HTTPException(status_code=404, detail="File not found")
    # print(model_name)
    filename = f"{model_name}_quantized.pt"
    return FileResponse(path=quantized_model, filename=filename)


# 显示前端页面
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
