#!/bin/bash

cd tumor_pred

echo "START UNCERTAINTY MODEL"

bash pred_thread_lym_con.sh \
    ../output 0 1 ../model/model_cons.t7 \
    >> ../logs_pred/log_uncertainty.txt 2>&1

echo "DONE"
