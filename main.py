from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import os
import shutil
from pathlib import Path
from quantization import quantize_model
from convert import convert_pytorch_to_onnx

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)  # 创建文件夹用于存储上传的文件
ONNX_DIR = "temp"

model_name = ''
model_type_g = ''


# 上传模型
@app.post("/quantize/")
async def quantize_model_endpoint(
        file: UploadFile = File(...),
        quantization_type: str = Form(...),  # int4 or int8
        model_type: str = Form(...),  # pytorch or tensorflow
        quantization_method: str = Form(...),
        quantization_method2: str = Form(...)
):
    global model_type_g
    model_type_g = model_type
    try:
        # 检查文件扩展名
        if not file.filename.endswith(".pt") and model_type == "pytorch":
            raise HTTPException(status_code=400, detail="Only .pt files are allowed for PyTorch models")
        elif not file.filename.endswith(".h5") and model_type == "tensorflow":
            raise HTTPException(status_code=400, detail="Only .h5 files are allowed for TensorFlow models")

        # 将上传的文件保存到服务器
        model_path = UPLOAD_FOLDER / file.filename
        global model_name  # 全局变量，保存模型名字
        model_name = os.path.splitext(file.filename)[0]
        with open(model_path, "wb") as buffer:
            buffer.write(await file.read())

        # 量化模型
        quantized_model_path, model_structure = quantize_model(model_path, quantization_type, model_type, model_name, quantization_method, quantization_method2)

        return {
            "quantized_model_path": str(quantized_model_path),
            "model_structure": model_structure
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantization failed: {e}")


# 下载量化后的模型
@app.get("/download/")
async def download_model(quantized_model: str):
    if not os.path.exists(quantized_model):
        raise HTTPException(status_code=404, detail="File not found")
    filename = ''
    if model_type_g == 'pytorch':
        filename = f"{model_name}_quantized.pt"
    elif model_type_g == 'tensorflow':
        filename = f"{model_name}_quantized.tflite"
    # print(filename)
    return FileResponse(path=quantized_model, filename=filename)

# 转换 PyTorch 模型为 ONNX
@app.post("/convert/")
async def convert_to_onnx(file: UploadFile = File(...)):
    # 保存上传的模型文件
    model_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 调用转换函数
    onnx_model_path = convert_pytorch_to_onnx(model_path, ONNX_DIR)

    return JSONResponse({
        "onnx_model_path": f"/download_onnx/?onnx_model={onnx_model_path}"
    })

# ONNX 模型下载
@app.get("/download_onnx/")
async def download_onnx_model(onnx_model: str):
    return FileResponse(onnx_model)


# 显示前端页面
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
