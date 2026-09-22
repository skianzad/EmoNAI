# Set FASTVLM_MLX_PATH for run_facs_tflite.py, mlx_vlm, etc.
# Usage:
#   source /Users/soheil/Documents/random_projects/ml-fastvlm/env-fastvlm-mlx.sh
# Optional: put that line in ~/.zshrc to persist.
#
# Default points at the 1.5B bundle from: app/get_pretrained_mlx_model.sh --model 1.5b --dest app/FastVLM/model

: "${FASTVLM_MLX_PATH:=/Users/soheil/Documents/random_projects/ml-fastvlm/app/FastVLM/model}"
export FASTVLM_MLX_PATH
