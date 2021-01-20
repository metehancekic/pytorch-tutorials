#!/bin/bash 

# model=VGG_modified
# frontend=Identity

# COMMAND="python -m pytorch-tutorials.cifar.play_ground  \
# --model=$model --epochs 100 --frontend=$frontend -at -tr"
# echo $COMMAND
# eval $COMMAND

model=VGG_modified
frontend=Identity

COMMAND="python -m pytorch-tutorials.cifar.main  \
--model=$model --epochs 100 --frontend=$frontend -at -an"
echo $COMMAND
eval $COMMAND

# model=VGG_modified2
# frontend=Identity

# COMMAND="python -m pytorch-tutorials.cifar.main  \
# --model=$model --epochs 100 --frontend=$frontend -at -an"
# echo $COMMAND
# eval $COMMAND

