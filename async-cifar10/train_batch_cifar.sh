#!/bin/sh

N_SERVER=4
LOSS=0%
# STDEV=0
# STALENESS_THRESHOLD=9999
mkdir -p logsCifar10YiBuWithNoNoise

# echo ""
# echo " ---- STARTING $LOSS $STDEV $STALENESS_THRESHOLD $i ---- "
# echo ""
# ./train_cifar.sh $N_SERVER $LOSS $STDEV $STALENESS_THRESHOLD 2>&1 | tee logsCifar10YiBuWithNoNoise/$LOSS-$STDEV-$STALENESS_THRESHOLD-$i.log


for STDEV in 0 1000; do
    for STALENESS_THRESHOLD in 1 9999; do
        echo ""
        echo " ---- STARTING $LOSS $STDEV $STALENESS_THRESHOLD $i ---- "
        echo ""
        ./train_cifar.sh $N_SERVER $LOSS $STDEV $STALENESS_THRESHOLD 2>&1 | tee logsCifar10YiBuWithNoNoise/$LOSS-$STDEV-$STALENESS_THRESHOLD-$i.log
    done
done


