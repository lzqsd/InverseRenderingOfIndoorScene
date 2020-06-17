# TEST
python3 testReal.py --cuda --dataRoot DATA --imList imList_20.txt \
    --testRoot Real20 --isLight --isBS --level 2 \
    --experiment0 check_cascade0_w320_h240 --nepoch0 14 \
    --experimentLight0 check_cascadeLight0_sg12_offset1 --nepochLight0 10 \
    --experimentBS0 checkBs_cascade0_w320_h240 \
    --experiment1 check_cascade1_w320_h240 --nepoch1 7 \
    --experimentLight1 check_cascadeLight1_sg12_offset1 --nepochLight1 10 \
    --experimentBS1 checkBs_cascade1_w320_h240 \
