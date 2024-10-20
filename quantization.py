import torch
import tensorflow as tf
from pathlib import Path


def quantize_model(model_path: Path, quantization_type: str, model_type: str, model_name: str):
    global save_path, model_structure
    if model_type == "pytorch":
        # 加载模型
        model = torch.jit.load(model_path)
        model.eval()

        # 根据选择的量化类型
        dtype = torch.qint8 if quantization_type == "int8" else torch.qint4

        # 进行量化
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=dtype
        )

        # 保存量化后的模型
        model.eval()  # 切换到评估模式
        scripted_model = torch.jit.script(model)  # 转换为 TorchScript
        save_path = f'D:/桌面/code/项目/model_quantization/temp/{model_name}_quantized.pt'
        # scripted_model.save(scripted_model, save_path)  # 保存模型
        scripted_model.save(f"./temp/{model_name}_quantized.pt")

        # 返回模型的结构
        model_structure = str(model)

    elif model_type == "tensorflow":
        model = tf.keras.models.load_model(model_path)

        # TensorFlow 模型量化 (假设基于int8/int4的量化)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantization_type == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int4]

        tflite_model = converter.convert()

        save_path = model_path.with_suffix(f"D:/桌面/code/项目/model_quantization/temp/{model_name}_quantized.tflite")
        with open(save_path, "wb") as f:
            f.write(tflite_model)

        # 返回模型的结构
        model_structure = model.summary(print_fn=lambda x: x)

    return save_path, model_structure


# print(quantize_model(Path('cnn_model.pt'),'int8','pytorch','cnn_model_qt'))
# print(quantize_model(Path('tf_model.h5'),'int8','tensorflow','tf_model'))
