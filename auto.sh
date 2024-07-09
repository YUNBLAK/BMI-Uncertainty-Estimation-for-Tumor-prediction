#!/bin/bash

echo "START EXTRACTING PATCHES"
bash save_svs_to_tiles.sh
echo "EXTRACTION - [DONE]"
echo ""

echo "START UNCERTAINTY PREDICTION"
bash start_con.sh
echo "UNCERTAINTY - [DONE]"
echo ""

echo "START BASELINE PREDICTION"
bash start.sh
echo "BASELINE - [DONE]"
echo ""