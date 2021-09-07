# Scale Pose 2D

## Intuition

Multi-scale training, multi-scale structure and multi-scale refinement. 

## Resources

[Pickle files](https://www.dropbox.com/sh/3wydln5k1xnfupc/AADa0Jnx_gkAhpelTYCnzMxLa?dl=0) for keypoints and segmentation.

## Installation 

Just install pytorch, install torchsul, and build the fastgaus. 

## Usage 

1. Generate the masks using dump.py (included in the resource link)

2. Train the network 
```
python distrib.py 
```
