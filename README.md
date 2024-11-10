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
添加了选项将量化和转换合到一起，增加了量化方法和对应的一些量化策略。  
主要针对pytorch模型进行改动，TensorFlow模型尚未实现  

![image](https://github.com/user-attachments/assets/af3bdd9e-7ec5-4d56-a2e9-66cb197dd20f)  

![image](https://github.com/user-attachments/assets/7cda1fe1-2882-4222-a3ce-82119ea40134)


