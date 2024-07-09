#!/bin/bash

# source ../../conf/variables.sh

echo "START"
FOLDER=$1
PARAL=$2
MAX_PARAL=$3
DEVICE=$4

echo "FOLDER: $FOLDER"
echo "PARAL: $PARAL"
echo "MAX_PARAL: $MAX_PARAL"
echo "DEVICE: $DEVICE"

DATA_FILE=prediction.txt
DONE_FILE=extraction_done.txt
EXEC_FILE=pred_con.py

PRE_FILE_NUM=0
REPEAT_COUNT=0
MAX_REPEAT=2

while [ ${REPEAT_COUNT} -lt ${MAX_REPEAT} ]; do
    LINE_N=0
    FILE_NUM=0
    EXTRACTING=0
    for files in ${FOLDER}/*; do
        echo "Checking folder: $files"
        
        FILE_NUM=$((FILE_NUM+1))
        if [ ! -f "${files}/${DONE_FILE}" ]; then 
            EXTRACTING=1
            echo "${files}/${DONE_FILE} not found, setting EXTRACTING to 1"
        fi
        echo "TEST $(basename "${files}" .svs)"
        LINE_N=$((LINE_N+1))
        NEWPATH="../prediction_values/$(basename "${files}" .svs)_prediction.txt"
        
        if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then 
            continue
        fi

        if [ -f "${files}/${DONE_FILE}" ]; then
            if [ ! -f "${NEWPATH}" ]; then
                echo ""
                echo ""
                echo ""
                echo "${NEWPATH} generating"
                python -u ${EXEC_FILE} ${files} ../model/ ${DATA_FILE} ../model/model_cons.t7
            fi
        fi
    done

    echo "EXTRACTING: $EXTRACTING, PRE_FILE_NUM: $PRE_FILE_NUM, FILE_NUM: $FILE_NUM"

    if [ ${EXTRACTING} -eq 0 ] && [ ${PRE_FILE_NUM} -eq ${FILE_NUM} ]; then 
        break
    fi
    PRE_FILE_NUM=${FILE_NUM}
    REPEAT_COUNT=$((REPEAT_COUNT + 1))
done
