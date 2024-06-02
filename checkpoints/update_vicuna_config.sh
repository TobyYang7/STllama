#!/bin/bash

# Define the file path
config_file="vicuna-7b-v1.5-16k/config.json"

# Define the new JSON content
new_content='{
  "_name_or_path": "vicuna-7b-v1.5-16k",
  "architectures": ["LlamaForCausalLM"],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_sequence_length": 16384,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-5,
  "rope_scaling": {
    "factor": 4.0,
    "type": "linear"
  },
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.34.0",
  "use_cache": true,
  "vocab_size": 32000,
  "st_hidden_size": 64,
  "lin_hidden_size": 128,
  "time_steps": 12,
  "pretrain_ST_model_path": "./checkpoints/st_encoder/pretrain_stencoder.pth"
}'

# Replace the content of the config file
echo "$new_content" > "$config_file"

# Confirm the change
if [ $? -eq 0 ]; then
  echo "The config file has been successfully updated."
else
  echo "An error occurred while updating the config file."
fi
