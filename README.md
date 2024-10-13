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
requirements.txt
```
目前只写了对pytorch模型的量化，量化方法用的也是pytorch中最简单的一种  
后续我打算在页面添加下拉框，能够传入不同的模型进行量化  

![image](https://github.com/user-attachments/assets/41169de8-8247-47bb-be8a-d9a7cf171428) 
