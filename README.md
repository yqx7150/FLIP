# FLIP

**Paper:** Undersampled FZA Lensless Imaging via Dual Diffusion Models with Intermediate Prior Learning

**Authors**: Yuxuan Zou, Ming Gao, Dingxiang Yuan, Nan Chen, Tianshui Yu, Wenbo Wan, Qiegen Liu, Senior Member, IEEE

Optics and Laser Technology [https://doi.org/10.1016/j.optlastec.2026.114665]

Date : Jan-22-2026  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2026, School of Information Engineering, Nanchang University. 

Lensless imaging replaces lens groups with planar encoding devices to effectively reduce the size of optical systems. Multi-phase Fresnel zone aperture (FZA) encoding effectively suppresses twin image artifacts but leads to field-of-view compression. To address the issue of image reconstruction under undersampling conditions, an undersampled FZA lensless imaging via dual diffusion models with intermediate prior learning (FLIP) is proposed. An intermediate-state high-dimensional tensor is constructed to exploit the prior association between the target images and the twin images from different phases. To achieve high-quality image reconstruction with network interpretability, the prior information obtained from the masked-sampling intermediate diffusion model is employed to constrain the recovery process of undersampled data, while the prior information obtained from the full-sampling intermediate diffusion model is employed to suppress artifacts in the reconstruction results. The measurement data are incorporated into the iterative process as a data-consistency term to ensure data consistency. Compared with the conventional methods, FLIP demonstrates significant improvements in the image recovery of undersampled data and effectively suppresses the artifacts caused by undersampling. The proposed method is capable of maintaining satisfactory data recovery capability even under the extreme condition of a 25% sampling rate.

## Main processes of lensless imaging. optical encoding and computational decoding.
<div align="center"><img src="https://github.com/yqx7150/FLIP/blob/main/Fig1.png"> </div>

## Schematic diagram of four-phase FZA lensless sampling. With the detector held fixed, the projection of a point source is completely sampled at (a) large distances and (b) intermediate distances, whereas at (c) short distances partial sampling leads to undersampling.
<div align="center">
    <img src="https://github.com/yqx7150/FLIP/blob/main/Fig2.png" style="width: 60%;">
</div>

## Four-phase lensless imaging. (a)Target image, (b)Measurement domain images of four phases, (c)Four intermediate images, (d) Reconstructed image.
<div align="center">
    <img src="https://github.com/yqx7150/FLIP/blob/main/Fig4.png" style="width: 100%;">
</div>

## Flowchart of FLIP.(a)Prior learning: masked-sampling intermediate diffusion model (MI-DM) captures gradient priors in undersampled regions, while full-sampling intermediate diffusion model (FI-DM) learns global structure. (b)Image reconstruction: the reconstruction process iteratively alternates between the MI-DM and the FI-DM to achieve accurate image recovery. (c)Data consistency: the measurement data are incorporated into the iterative process as a data-consistency term to ensure data consistency.
<div align="center"><img src="https://github.com/yqx7150/FLIP/blob/main/Fig3.png"> </div>

## Reconstruction on the LSUN-Church dataset. (a) GT, (b) SP-BP, (c) BP, (d) CS, (e) ADMM, (f)MIRNet-v2, (g) MLDM_I, (h) FLIP.
<div align="center"><img src="https://github.com/yqx7150/FLIP/blob/main/Fig5.png"> </div>

## Reconstructed images under different imaging distances. (a) GT, (b) SP-BP, (c) BP, (d) CS, (e) ADMM, (f) MIRNet-v2, (g) MLDM_I, (h) FLIP.
<div align="center"><img src="https://github.com/yqx7150/FLIP/blob/main/Fig6.png"> </div>

## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26

## Checkpoints

MI-DM:We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud](https://pan.baidu.com/s/1bYIXgJ7Rno951dvIh75xVw?pwd=n5xd) Extract the code (hdtp)

FI-DM:We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud](https://pan.baidu.com/s/14-5coCKB3Z3vTH4--QvQuw?pwd=9q7s) Extract the code (hdtp)

## Dataset
The dataset used to train the model in this experiment is  LSUN-bedroom and  LSUN-church.

## Train:

### MI-DM

python main.py --config=configs/ve/church_ncsnpp_continuous_qian12tongdao.py  --workdir=exp_train_church_UIDM --mode=train --eval_folder=result

### FI-DM

python main.py --config=configs/ve/church_ncsnpp_continuous_qian12tongdao.py  --workdir=exp_train_church_FIDM --mode=train --eval_folder=result

## Test:

Simulate : python FLIP_reconstruction_simulate.py

Experiment : python FLIP_reconstruction_experiment.py


## Acknowledgement
The implementation is based on this repository: https://github.com/yang-song/score_sde_pytorch.

