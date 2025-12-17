import os
import json
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from ai_edge_torch.generative.utilities import converter
try:
    from ai_edge_torch.generative.utilities.export_config import ExportConfig
except ImportError:
    try:
        from ai_edge_torch.generative.export_config import ExportConfig
    except ImportError:
        ExportConfig = None
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.examples.gemma3.decoder import Decoder
from ai_edge_torch.generative.utilities.loader import ModelLoader

# 1. Configuration
MODEL_ID = "google/medgemma-4b-it"
OUTPUT_PATH = "./medgemma_multimodal_16k"
MODEL_NAME = "medgemma_16k_int8"
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

# 2. Load MedGemma text config from Hugging Face
print("Fetching MedGemma-4b-it config.json...")
config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=HF_TOKEN)
with open(config_path, "r") as f:
    cfg_json = json.load(f)
text_cfg = cfg_json["text_config"]

# Extract key parameters
num_layers = int(text_cfg["num_hidden_layers"])  # 34
num_heads = int(text_cfg["num_attention_heads"])  # 8
num_kv_heads = int(text_cfg["num_key_value_heads"])  # 4
head_dim = int(text_cfg["head_dim"])  # 256
hidden_size = int(text_cfg["hidden_size"])  # 2560
intermediate_size = int(text_cfg["intermediate_size"])  # 10240
vocab_size = int(text_cfg["vocab_size"])  # 262208
max_pos = int(text_cfg["max_position_embeddings"])  # 131072
rope_theta = int(text_cfg["rope_theta"])  # 1000000
rope_local_base = int(text_cfg.get("rope_local_base_freq", 10000))
sliding_window = int(text_cfg.get("sliding_window", 1024))
layer_types = text_cfg["layer_types"]  # list of "sliding_attention" / "full_attention"

# 3. Build AI Edge Torch ModelConfig
norm = cfg.NormalizationConfig(
    type=cfg.NormalizationType.RMS_NORM,
    epsilon=1e-6,
    zero_centered=True,
)
ff_conf = cfg.FeedForwardConfig(
    type=cfg.FeedForwardType.GATED,
    activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
    intermediate_size=intermediate_size,
    pre_ff_norm_config=norm,
    post_ff_norm_config=norm,
)

block_confs = []
for i in range(num_layers):
    attn_type = cfg.AttentionType.GLOBAL if layer_types[i] == "full_attention" else cfg.AttentionType.LOCAL_SLIDING
    attn_conf = cfg.AttentionConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        num_query_groups=num_kv_heads,
        rotary_base=(rope_theta if attn_type == cfg.AttentionType.GLOBAL else rope_local_base),
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        qkv_use_bias=False,
        output_proj_use_bias=False,
        query_norm_config=norm,
        key_norm_config=norm,
        logit_softcap=None,
        sliding_window_size=sliding_window,
        attn_type=attn_type,
    )
    block_confs.append(
        cfg.TransformerBlockConfig(
            attn_config=attn_conf,
            ff_config=ff_conf,
            pre_attention_norm_config=norm,
            post_attention_norm_config=norm,
        )
    )

model_conf = cfg.ModelConfig(
    vocab_size=vocab_size,
    num_layers=num_layers,
    max_seq_len=max_pos,
    embedding_dim=hidden_size,
    block_configs=block_confs,
    final_norm_config=norm,
    embedding_scale=hidden_size ** 0.5,
    lm_head_use_bias=False,
    lm_head_share_weight_with_embedding=True,
    final_logit_softcap=None,
)

# 4. Build Decoder model and load weights from HF checkpoint
print("Snapshotting MedGemma-4b-it weights...")
local_dir = snapshot_download(
    repo_id=MODEL_ID,
    token=HF_TOKEN,
    allow_patterns=["*.safetensors", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
)

# Define tensor name mappings that match MedGemma weight keys
tensor_names = ModelLoader.TensorNames(
    ff_up_proj="language_model.model.layers.{}.mlp.up_proj",
    ff_down_proj="language_model.model.layers.{}.mlp.down_proj",
    ff_gate_proj="language_model.model.layers.{}.mlp.gate_proj",
    attn_query_proj="language_model.model.layers.{}.self_attn.q_proj",
    attn_key_proj="language_model.model.layers.{}.self_attn.k_proj",
    attn_value_proj="language_model.model.layers.{}.self_attn.v_proj",
    attn_output_proj="language_model.model.layers.{}.self_attn.o_proj",
    attn_query_norm="language_model.model.layers.{}.self_attn.q_norm",
    attn_key_norm="language_model.model.layers.{}.self_attn.k_norm",
    pre_attn_norm="language_model.model.layers.{}.input_layernorm",
    post_attn_norm="language_model.model.layers.{}.post_attention_layernorm",
    pre_ff_norm="language_model.model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="language_model.model.layers.{}.post_feedforward_layernorm",
    embedding="language_model.model.embed_tokens",
    embedding_position=None,
    final_norm="language_model.model.norm",
    lm_head=None,
)

print("Building AI Edge decoder and loading weights...")
edge_decoder = Decoder(model_conf, mask_cache_size=0)
loader = ModelLoader(local_dir, tensor_names)
missing, unexpected = loader.load(edge_decoder, strict=not model_conf.lm_head_share_weight_with_embedding)
print("Loaded weights. Missing:", missing, "Unexpected:", unexpected)
# Reduce parameter memory footprint
edge_decoder = edge_decoder.to(torch.bfloat16)
edge_decoder.eval()

# 5. Export to TFLite using AI Edge converter
export_cfg = ExportConfig() if ExportConfig else None
if export_cfg:
    export_cfg.mask_as_input = True
    export_cfg.decode_batch_size = 1

# Monkeypatch KV cache to use bfloat16 to reduce memory and match CPU SDPA
from ai_edge_torch.generative.layers import kv_cache as kv_utils
_original_from_model_config = kv_utils.KVCache.from_model_config

def _from_model_config_bf16(kv_cache_max_len, config, dtype=torch.bfloat16, device=None, batch_size=1, kv_layout=kv_utils.KV_LAYOUT_DEFAULT):
    return _original_from_model_config(kv_cache_max_len, config, dtype=dtype, device=device, batch_size=batch_size, kv_layout=kv_layout)

kv_utils.KVCache.from_model_config = _from_model_config_bf16

print("Converting to TFLite with reduced context (prefill=256, kv=4096) to fit memory...")
converter.convert_to_tflite(
    edge_decoder,
    output_path=OUTPUT_PATH,
    output_name_prefix=MODEL_NAME,
    prefill_seq_len=256,
    kv_cache_max_len=4096,
    quantize="dynamic_int8",
    export_config=export_cfg,
)

print(f"Conversion complete: {OUTPUT_PATH}/{MODEL_NAME}.tflite")
