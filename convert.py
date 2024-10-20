import torch
import torch.onnx as torch_onnx
import os
import onnx


def convert_pytorch_to_onnx(model_path: str, output_dir: str):

    # 加载 PyTorch 模型
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    # 生成 ONNX 模型保存路径
    onnx_model_path = os.path.join(output_dir, f"{os.path.basename(model_path).split('.')[0]}.onnx")

    # 模拟的输入数据，适用于 MNIST 模型
    dummy_input = torch.randn(1, 1, 28, 28)

    # 导出 ONNX 模型
    torch_onnx.export(model, dummy_input, onnx_model_path, export_params=True)

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    return onnx_model_path
