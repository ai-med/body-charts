from typing import Dict, Any

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llama-3.3-awq-kosbu-fp8kv": {
        "default_model": "kosbu/Llama-3.3-70B-Instruct-AWQ",
        "max_model_len": 3072,
        "max_num_seqs": 2,
        "tensor_parallel_size": 1,
        "quantization": "awq_marlin",
        "kv_cache_dtype": "fp8",
        "enable_prefix_caching": True,
    },
    "qwen3-32b-awq": {
        "default_model": "Qwen/Qwen3-32B-AWQ",
        "max_model_len": 3072,
        "max_num_seqs": 4,
        "tensor_parallel_size": 1,
        "quantization": "awq_marlin",
        "kv_cache_dtype": "fp8",
        "enable_prefix_caching": True,
    },
    "medgemma-27b-text-it-gguf": {
        "default_model": "unsloth/medgemma-27b-text-it-GGUF",
        "max_model_len": 4096,         
        "max_num_seqs": 4,             
        "tensor_parallel_size": 1,     
        "quantization": "gguf",        
        "kv_cache_dtype": "fp8"        
    },
    "medgemma": {
        "default_model": "google/medgemma-27b-text-it",
        "max_model_len": 3500,
        "max_num_seqs": 2,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,       # helpful when prompts are identical
    },
    "openbiollm-70b-awq-fp8kv": {
        "default_model": "TitanML/Llama3-OpenBioLLM-70B-AWQ-4bit",
        "max_model_len": 8192,
        "max_num_seqs": 2,
        "tensor_parallel_size": 1,
        "quantization": "awq_marlin",
        "kv_cache_dtype": "fp8",
        "enable_prefix_caching": True
    },
    "qwen25-72b-awq": {
        "default_model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "max_model_len": 8192,
        "max_num_seqs": 2,
        "tensor_parallel_size": 1,
        "quantization": "awq_marlin",
        "kv_cache_dtype": "fp8",
        "enable_prefix_caching": True
        }
} 
