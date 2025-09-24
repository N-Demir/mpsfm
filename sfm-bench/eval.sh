#!/bin/bash
set -e

# Check if output_folder argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_folder>"
    echo "Example: $0 /sfm-bench/methods/map-anything/mipnerf360/bicycle"
    exit 1
fi
output_folder=$1

######## START OF YOUR CODE ########

python reconstruct.py \
    --conf sp-lg_m3dv2 \
    --data_dir $output_folder \
    --refrec_dir $output_folder/sparse_gt/0
    # --images_dir $output_folder/images