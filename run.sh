#!/usr/bin/sh

python3 src/run_AURC_token.py \
    --card_number=1 \
    --train \
    --crf \
    --target_domain='In-Domain' \
#    --target_domain='Cross-Domain' \
