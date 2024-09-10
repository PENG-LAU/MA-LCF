# MA-LCF: Multi-Angle Linked Co-Fusion Network For 3D Human Pose Estimation On Monocular Video Stream

A 3D HPE Method In Monocular Video.
<!-- ***
![skating](demo/view/demo1.gif) -->
*** 

## Install
`pip install -r requirements.txt`

## Dataset
### Human 3.6M
Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed Human 3.6M dataset [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'. Then, slice the motion clips by running the following python code in "data/preprocess" directory:

```bash
# set up you need frames , for example: 243 / 81 / 27.
python h36m.py  --n-frames 81
```
### MPI-INF-3DHP
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. And the generated ".npz" files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

## Train
to train a model in Human 3.6M dataset:
``` bash
python train.py --config configs/h36m/yourmodel.yaml
# before trian gt model, you need to set use_porj_as_2d = Ture.
python train.py --config configs/h36m/yourmodel.yaml
```
to train a model in MPI-INF-3DHP dataset:
``` bash
python train_3dhp.py --config configs/mpi/yourmodel.yaml
```

## Test
to test a model in Human 3.6M Dataset:
``` bash
# both cpn model or gt model use this command.
python train.py --eval-only --checkpoint checkpoint --checkpint-file  yourfilename  --config configs/h36m/yourmodel.yaml
```
to test a model in MPI-INF-3DHP Dataset:
``` bash
python train_3dhp.py --eval-only --checkpoint mpi-checkpoint --checkpoint-file yourfilename --config configs/mpi/yourmodel.yaml
```
## Visualization
Run the following command in the `data/preprocess` directory:
```text
python visualize.py --dataset h36m --sequence-number <AN ARBITRARY NUMBER>
```
This should create a gif file at `data` directory. MPI-INF-3DHP dataset use same command, but `--dataset` should be set to `mpi`.
## Demo
First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory.  Then, put your videos in the './demo/video' directory. Last, run the command below:
```
python demo/vis.py --video sample_video.mp4
```
## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)

