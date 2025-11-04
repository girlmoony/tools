import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import os

def get_intermediate_outputs_model(model_path, output_names):
    """
    指定された中間層を出力に追加した一時的なONNXモデルを生成し、パスを返します。
    """
    model = onnx.load(model_path)
    # 既存の出力をセットに格納
    existing_outputs = {o.name for o in model.graph.output}
    
    for output_name in output_names:
        if output_name not in existing_outputs:
            # グラフの全value_infoから対応する情報を探す
            found_info = None
            for info in model.graph.value_info:
                if info.name == output_name:
                    found_info = info
                    break
            for output_val in model.graph.output:
                if output_val.name == output_name:
                    found_info = output_val
                    break
            # ノードの出力からも探す
            if found_info is None:
                for node in model.graph.node:
                    for output in node.output:
                        if output == output_name:
                            # Shape情報を推論で取得する（手動でshape指定してもOK）
                            # onnx.shape_inference.infer_shapes(model) を事前に実行しておくのが望ましい
                            found_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)
                            break
            
            if found_info:
                model.graph.output.append(found_info)
                print(f"Added '{output_name}' to model outputs.")
            else:
                print(f"Warning: Could not find ValueInfo for '{output_name}'. Cannot add as output.")


    # 一時ファイルとして保存
    temp_model_path = model_path.replace(".onnx", "_temp_outputs.onnx")
    onnx.save(model, temp_model_path)
    return temp_model_path

def run_inference_and_get_tensors(model_path, input_data, output_names):
    """
    ONNXモデルで推論を実行し、指定された出力テンソルを辞書形式で返します。
    """
    # 修正した一時モデルを使用
    temp_model_path = get_intermediate_outputs_model(model_path, output_names)
    
    sess_options = ort.SessionOptions()
    # CPU Execution Providerを使用（iPro環境によってはここを調整）
    session = ort.InferenceSession(temp_model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
    
    # 入力名を取得
    input_name = session.get_inputs()[0].name
    
    # 推論実行
    # run([出力名リスト], {入力名: 入力データ})
    results = session.run(output_names, {input_name: input_data})
    
    # 一時ファイルを削除
    os.remove(temp_model_path)
    
    return dict(zip(output_names, results))

def compare_tensors(tensor_fp32, tensor_quantized, layer_name):
    """
    2つのテンソルの差異（MAE, RMSE, コサイン類似度）を計算します。
    """
    print(f"\n--- Comparing Layer: {layer_name} ---")
    # フラット化して比較
    fp32_flat = tensor_fp32.flatten()
    quant_flat = tensor_quantized.flatten()
    
    mae = np.mean(np.abs(fp32_flat - quant_flat))
    rmse = np.sqrt(np.mean((fp32_flat - quant_flat)**2))
    
    # コサイン類似度（分布の形状比較に便利）
    norm_fp32 = np.linalg.norm(fp32_flat)
    norm_quant = np.linalg.norm(quant_flat)
    if norm_fp32 != 0 and norm_quant != 0:
        cosine_similarity = np.dot(fp32_flat, quant_flat) / (norm_fp32 * norm_quant)
    else:
        cosine_similarity = "Undefined (zero vector)"

    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"  Cosine Similarity: {cosine_similarity}")
    
    return mae, rmse, cosine_similarity

# ==============================================
# 実行部分
# ==============================================

# 1. モデルパスと入力データの準備
FP32_MODEL_PATH = "model_fp32.onnx" # FP32モデルのパスを指定してください
QUANTIZED_MODEL_PATH = "model_quantized.onnx" # 量子化モデルのパスを指定してください

# EfficientNetの入力形状に合わせてダミー入力データを作成 (例: バッチサイズ1, 3チャンネル, 224x224画像)
# 実際のキャリブレーション画像の一つをここに入れるのがベスト
input_shape = (1, 3, 224, 224) 
dummy_input = np.random.randn(*input_shape).astype(np.float32) 
# または実際の画像読み込み:
# image = load_image_function("path/to/sushi_image.jpg")
# dummy_input = preprocess_image(image)


# 2. 比較したい中間層のテンソル名をリストアップ
# テンソル名はNetronなどのONNXビューアツールで確認できます
# 例: EfficientNetの場合、'blocks_1a_output', 'blocks_3b_output', 'head_swish'など
TARGET_LAYER_NAMES = [
    'output_of_some_early_layer',
    'output_of_middle_layer',
    'output_of_penultimate_layer',
    'output_of_final_logit_layer' # 最終出力も含める
]


# 3. 各モデルから中間層の結果を取得
print("Running FP32 inference...")
tensors_fp32 = run_inference_and_get_tensors(FP32_MODEL_PATH, dummy_input, TARGET_LAYER_NAMES)

print("\nRunning Quantized inference...")
tensors_quantized = run_inference_and_get_tensors(QUANTIZED_MODEL_PATH, dummy_input, TARGET_LAYER_NAMES)


# 4. 結果の比較
print("\n--- Summary of Differences ---")
for layer_name in TARGET_LAYER_NAMES:
    if layer_name in tensors_fp32 and layer_name in tensors_quantized:
        compare_tensors(tensors_fp32[layer_name], tensors_quantized[layer_name], layer_name)
    else:
        print(f"Could not compare layer {layer_name}. Check if the name is correct or if it was added as output.")

