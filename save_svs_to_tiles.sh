#!/bin/bash

#source ../conf/variables.sh

echo "START"
COD_PARA=0
MAX_PARA=1
IN_FOLDER=dataset                                                   # data/svs
OUT_FOLDER=output                                                   # data/patches

LINE_N=0
for files in ${IN_FOLDER}/*.*; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARA)) -ne ${COD_PARA} ]; then continue; fi

    SVS=`echo ${files} | awk -F'/' '{print $NF}'`

    echo "START EXTRACTING"
    python save_svs_to_tiles.py ${SVS} ${IN_FOLDER} ${OUT_FOLDER}
    
    if [ $? -ne 0 ]; then
        echo "failed extracting patches for " ${SVS}
        rm -rf ${OUT_FOLDER}/${SVS}
    else
        touch ${OUT_FOLDER}/${SVS}/extraction_done.txt
    fi

done

exit 0;