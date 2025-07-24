#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python3 main.py \
python main.py \
 --datasetName 'VimeoSepTuplet' \
 --modelName 'base3GAP' \
 --test_batch_size 16 \
 --max_epoch 200 \
 --lr 0.0000001 \
 --resume 1 \
 --checkpoint_dir "D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\Results\20250316_1436_74976\base3GAP_epoch199.pth" \
 --save_images True \
 --test_custom 1 \
 --scale 8
#
# python main.py `
# --datasetName 'VimeoSepTuplet' `
# --modelName 'base3GAP' `
# --test_batch_size 16 `
# --max_epoch 200 `
# --lr 0.0000001 `
# --resume 1 `
# --checkpoint_dir "D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\Results\20250316_1436_74976\base3GAP_best.pth" `
# --save_images True `
# --test_custom 1 `
# --scale 8
