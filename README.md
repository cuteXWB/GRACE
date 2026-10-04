# GRACE

Gradient-driven Robust Augmentation and Consistency for Generalizable Medical Image Segmentation.

## Parameters

Frozen UniMedCLIP ViT-B/16; Adam; training batch size 24; initial learning rate 3e-4; 100 epochs (EUS: 10); K=2; M=3; lambda_cons=1; lambda_virt=0.5; lambda_clip=0.10; seed=666. Testing uses batch size 32, sigmoid threshold 0.5, nearest-neighbor mask resizing and a two-pixel NSD tolerance. GVOS and PGC are disabled during inference.

## Setup

Use Python 3.10. Install PyTorch and torchvision compatible with CUDA, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Place UniMedCLIP weights at `checkpoints/unimed_clip_vit_b16.pt`. Place images and masks under `data/<dataset>/<Train_Folder|Val_Folder|Test_Folder>/img/` and `label/`. Configure paths in `configs/`. Data and pretrained weights are not included.

Place prompt spreadsheets under `data/<dataset>/Prompts_Folder/`, with columns `Image`, `Ground Truth` and `Description`. Training uses `Train_text.xlsx` and `Val_text.xlsx`, or `Train_text_<percent>.xlsx` and `Val_text_<percent>.xlsx` for annotation subsets. Testing uses `Test_text_original.xlsx`.

## Run

Single GPU, BUSI ID/OOD experiments:

```bash
bash scripts/run_GRACE_id_ood.sh \
  --seed 666 --gpu 0 --source-dataset BUSI \
  --percents 10,25,50,100 \
  --ood-targets BUSBRA,BUSUC,BUID,UDIAT \
  --ood-source-percent 100 \
  --output-root output_GRACE_BUSI_seed666 \
  --pretrained-path checkpoints/unimed_clip_vit_b16.pt
```

All source datasets and corresponding ID/OOD experiments on four GPUs:

```bash
bash run_GRACE_4gpu.sh \
  --seed 666 --gpus 0,1,2,3 \
  --output-root output_GRACE_all_seed666 \
  --pretrained-path checkpoints/unimed_clip_vit_b16.pt
```

Checkpoints, predicted masks and metrics are saved under the output directory. Repeating the command resumes incomplete training and skips completed stages.
