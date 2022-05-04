#!/bin/bash

CNNlr=(0.1 0.05)
LClr=(0.01 0.005)

for cnnlr in "${CNNlr[@]}"; do
for lclr in "${LClr[@]}"; do

    name="cnnlr=${CNNlr}-lclr=${lclr}"
    sbatch job.sh ${cnnlr} ${lclr}

done
done
