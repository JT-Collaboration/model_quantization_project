# model_quantization_project  

# 项目结构
```bash
static
    --index.css
templates  # 存放前端页面
    --index.html
temp  # 存放量化后的模型文件
    --model_quantized.pt
uploads  # 存放用户上传的模型文件
    --model.pt
main.py
quantization.py
convert.py
requirements.txt
```
对tensorflow的量化还有点问题，只写了pytorch转onnx格式

![image](https://github.com/user-attachments/assets/78c33751-b934-4bd7-ba6d-15999a9db37f)

