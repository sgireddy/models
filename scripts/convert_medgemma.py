import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig


# 1. Configuration
model_id = "google/medgemma-4b-it"
output_path = "./medgemma_multimodal_16k"
model_name = "medgemma_16k_int8"


# 2. Load the original base model
print("Loading MedGemma-4b-it...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# 3. Enhanced Export Configuration
# Higher KV cache max length to support medical image arrays (CT/MRI slices)
export_config = ExportConfig()
export_config.mask_as_input = True


print("Converting to TFLite with 16k context...")
converter.convert_to_tflite(
    model,
    output_path=output_path,
    output_name_prefix=model_name,
    # Supports ~2 images + initial prompt in the 'prefill' phase
    prefill_seq_len=1024,
    # Total context window for long medical reports or multiple image slices
    kv_cache_max_len=16384,
    # Weight-only Int8 quantization to keep the model under 4GB
    quantize="dynamic_int8",
    export_config=export_config,
)

print(f"Conversion complete: {output_path}/{model_name}.tflite")
