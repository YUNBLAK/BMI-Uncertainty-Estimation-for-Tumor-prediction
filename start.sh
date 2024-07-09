#!/bin/bash

cd tumor_pred

echo "START BASELINE MODEL"

bash pred_thread_lym.sh \
    ../output 0 1 ../model/model_base.t7 \
    >> ../logs_pred/log_base.txt 2>&1

echo "DONE"
