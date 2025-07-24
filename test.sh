#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
"D:\ANH\VFI_testdata\Slowmo\New folder (2)" \
--checkpoint "D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\Results\20250316_1436_74976\base3GAP_best.pth" \
--output_dir='D:\ANH\VFI_testdata\Slowmo\New folder (2)' \
--input_type directory \
--batch 8 \
--scale=4

#python test.py "D:\ANH\VFI_testdata\Slowmo\New folder (2)" --checkpoint "D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\Results\20250316_1436_74976\base3GAP_best.pth" --output_dir='D:\ANH\VFI_testdata\Slowmo\New folder (2)' --input_type directory --batch 8 --scale=4

--checkpoint_dir "D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\Results\20250316_1436_74976\base3GAP_best.pth"
--datasetPath "D:\KIEN\Dataset\vimeo_septuplet"
--modelName "base3GAP"
python quick_ssim_test.py --checkpoint_dir "D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\Results\20250316_1436_74976\base3GAP_best.pth" --datasetPath "D:\KIEN\Dataset\vimeo_septuplet" --modelName base3GAP