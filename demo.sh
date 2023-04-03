LOGFILE=loggers/${1}.log

CUDA_VISIBLE_DEVICES=5 python3 demo.py # > "$LOGFILE" 2>&1 &