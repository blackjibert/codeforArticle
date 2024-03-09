#!/bin/sh

N_SERVER=4
LOSS=0%
STDEV=0
STALENESS_THRESHOLD=1000 #yibu
mkdir -p logsFmnistMyWithNoise
#for i in {1..3}; do # Redo 3 times
#for LOSS in 0% 10% 20%; do
#for STDEV in 0 1000; do
#  for STALENESS_THRESHOLD in 0 9999; do
echo ""
echo " ---- STARTING $LOSS $STDEV $STALENESS_THRESHOLD $i ---- "
echo ""
./train_fmnist.sh $N_SERVER $LOSS $STDEV $STALENESS_THRESHOLD 2>&1 | tee logsFmnistMyWithNoise/$LOSS-$STDEV-$STALENESS_THRESHOLD-$i.log
# done
#done
#done
#done