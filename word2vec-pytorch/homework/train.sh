#! /bin/bash

LOG=$(realpath train.log)
CONFIG_FILE=$(realpath ../config.yaml)
SRC=$(realpath ../train.py)
TEMP_CONFIG=$(mktemp)

trap "rm -f $TEMP_CONFIG" 0 2 3 15

cd ..

for learning_rate in 0.025 0.05; do
    for train_batch_size in $(seq 32 32 160); do
        echo "learning_rate = $learning_rate; train_batch_size = $train_batch_size" >> $LOG
        yq e ".learning_rate = $learning_rate | .train_batch_size = $train_batch_size" $CONFIG_FILE > $TEMP_CONFIG
        python $SRC --config $TEMP_CONFIG | tee -a $LOG
        echo "" >> $LOG
    done
done
