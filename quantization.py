import torch
from pathlib import Path


def quantize_model(model_path: Path, model_name: str) -> str:
    # 加载模型
    model = torch.jit.load(model_path)
    model.eval()

    # 进行量化
    model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # 保存量化后的模型
    model.eval()  # 切换到评估模式
    scripted_model = torch.jit.script(model)  # 转换为 TorchScript
    save_path = f'D:/桌面/code/项目/model_quantization/temp/{model_name}_quantized.pt'
    # scripted_model.save(scripted_model, save_path)  # 保存模型
    scripted_model.save(f"./temp/{model_name}_quantized.pt")

    return save_path




'''# 2. 量化模型
def quantize_model(model):
    # 设置模型为评估模式
    model.eval()

    # 量化准备
    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.qconfig = qconfig

    # 准备量化
    torch.quantization.prepare(model, inplace=True)

    # 使用随机生成的输入进行校准
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST 图片的尺寸
    with torch.no_grad():
        model(dummy_input)  # 通过模型运行以收集量化统计信息

    # 转换为量化模型
    torch.quantization.convert(model, inplace=True)
    return model


quantized_model = quantize_model(loaded_model)

# 3. 保存量化模型
quantized_model.save('mnist_cnn_quantized.pt')
print("量化模型已保存为 mnist_cnn_quantized.pt")'''
