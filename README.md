# RAFT Sintel-only training notes

This fork adds a minimal `sintel_scratch` training path for training RAFT from random initialization using only MPI-Sintel clean/final frames and Sintel flow ground truth. It does not use FlyingChairs, FlyingThings3D, KITTI, or HD1K.

## Dataset expected by this setup

Download MPI-Sintel optical-flow training data and place it as:

```text
RAFT-master/
└── datasets/
    └── Sintel/
        └── training/
            ├── clean/
            ├── final/
            └── flow/
```

You do not need Chairs, Things, KITTI, or HD1K for `--stage sintel_scratch`.

## Training command

Linux/WSL:

```bash
bash train_sintel_scratch.sh
```

Equivalent explicit command:

```bash
python -u train.py \
  --name raft-sintel-scratch-bs4 \
  --stage sintel_scratch \
  --validation sintel \
  --gpus 0 \
  --num_steps 100000 \
  --batch_size 4 \
  --lr 0.0001 \
  --image_size 368 768 \
  --wdecay 0.00001 \
  --gamma 0.85 \
  --mixed_precision
```

Windows PowerShell:

```powershell
python -u train.py `
  --name raft-sintel-scratch-bs4 `
  --stage sintel_scratch `
  --validation sintel `
  --gpus 0 `
  --num_steps 100000 `
  --batch_size 4 `
  --lr 0.0001 `
  --image_size 368 768 `
  --wdecay 0.00001 `
  --gamma 0.85 `
  --mixed_precision
```

Do not pass `--restore_ckpt` if the goal is training from scratch.

## Changes made

### `core/datasets.py`

Added a new `sintel_scratch` stage inside `fetch_dataloader(args, TRAIN_DS='C+T+K+S+H')`.

```diff
+    elif args.stage == 'sintel_scratch':
+        # Sintel-only training from random initialization.
+        # This excludes FlyingThings3D, KITTI, and HD1K.
+        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
+        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
+        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
+        train_dataset = 100*sintel_clean + 100*sintel_final
+
     elif args.stage == 'sintel':
         aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
         things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
```

Reason: the original `--stage sintel` is not Sintel-only. It mixes Sintel with FlyingThings3D and, depending on `TRAIN_DS`, also KITTI and HD1K. The new stage leaves the original RAFT behavior untouched while adding a clean Sintel-only route.

### `train.py`

Changed BatchNorm freezing so it only happens when fine-tuning from a checkpoint.

```diff
-    if args.stage != 'chairs':
-        model.module.freeze_bn()
+    # If training from scratch directly on Sintel, do not freeze randomly initialized BN statistics.
+    # Keep original RAFT behavior only when fine-tuning from an existing checkpoint.
+    if args.stage != 'chairs' and args.restore_ckpt is not None:
+        model.module.freeze_bn()
```

And after validation:

```diff
-                if args.stage != 'chairs':
-                    model.module.freeze_bn()
+                if args.stage != 'chairs' and args.restore_ckpt is not None:
+                    model.module.freeze_bn()
```

Reason: the original RAFT training schedule trains on Chairs/Things first, so BatchNorm statistics already exist before Sintel fine-tuning. When training directly from scratch on Sintel, freezing BatchNorm immediately would freeze untrained/random running statistics.

### `train_sintel_scratch.sh`

Added a convenience launcher:

```diff
+#!/bin/bash
+mkdir -p checkpoints
+
+python -u train.py \
+  --name raft-sintel-scratch-bs4 \
+  --stage sintel_scratch \
+  --validation sintel \
+  --gpus 0 \
+  --num_steps 100000 \
+  --batch_size 5 \
+  --lr 0.0001 \
+  --image_size 368 768 \
+  --wdecay 0.00001 \
+  --gamma 0.85 \
+  --mixed_precision
```

Reason: this gives a reproducible single-GPU command for a 20 GB GPU. If CUDA memory fails, reduce `--batch_size 5` to `--batch_size 2`. If memory is available, try `--batch_size 5`.

---

# RAFT
This repository contains the source code for our paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Demos
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
