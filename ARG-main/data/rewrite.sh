
CUDA_VISIBLE_DEVICES=0,4,6,7 nohup python rewrite.py > "log_$(date +%Y%m%d_%H%M%S).log" 2>&1 &