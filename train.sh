LOGFILE=loggers/${1}.log

python3 train.py > "$LOGFILE" 2>&1 &