LOGFILE=loggers/${1}.log

CUDA_VISIBLE_DEVICES=4 python3 train.py > "$LOGFILE" 2>&1 &