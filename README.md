# OSRGAN-master

<!-- This repository is an official implementation of "[Orthogonal Subspace Representation Generative
Adversarial Networks](https://arxiv.org/abs/2110.00788v1)". -->
This repository is an official implementation of "Orthogonal Subspace Representation Generative
Adversarial Networks".



## Installation
Install CUDA 10.1 and Python 3.8.0 firstly, and clone the repository locally:
```shell
git clone https://github.com/XiangTodayEatsWhat/Inference-InfoGAN.git
```
Then, install PyTorch and torchvision:
```shell
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Install other requirements:
```
tqdm
scikit-learn
visdom
```

## Datasets

### dsprites dataset
Download [dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz](https://github.com/deepmind/dsprites-dataset) and create the file structure as follows:
```
└── data
    └── dSprites
        └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
└── OSRGAN-dSprites
```

### CelebA dataset
Download [img_align_celeba_png.7z](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and create the file structure as follows:
```
└── data
    └── CelebA
        └── img_align_celeba_png
            └── *.png
└── OSRGAN-CelebA
```

## Usage

### Initialize visdom:
```shell
python -m visdom.server
```
### Training
To train OSRGAN on dSprites with single gpu:
```shell
cd OSRGAN-dSprites
python train.py --model_type OBE
```
To train OSRGAN on CelebA with single gpu:
```shell
cd OSRGAN-CelebA
python train.py --model_type OBE
```

We also provide OSRGAN-DCT for training, just simply changing the model_type to DCT as follows:
```shell
python train.py --model_type DCT
```

### Evaluation
To evaluate OSRGAN on dSprites:
```shell
cd OSRGAN-dSprites
python test.py --model_type OBE --ckpt_load <ckpt_name>
```
Taking the [pre-trained model](https://drive.google.com/drive/folders/17GbsSJApws8GgVpUN2J8gMfpM5CUyfrV) as an example, the following output can be obtained:
```
factorVAE_metric:0.956, SAP_metric:0.6667119252548017, MIG_metric:0.5072938550840046 
```

To evaluate OSRGAN on CelebA:
```shell
cd OSRGAN-CelebA
python test.py --model_type OBE --ckpt_load <ckpt_name>
```

Visualization results can be checked on the visdom server http://localhost:8097/.

## Results
### dSprites
<p align="center">
<img src=figs/dsprites.png width="350">
</p>

### CelebA
<p align="center">
<img src=figs/celebA_1.png width="300"> <img src=figs/celebA_2.png width="300">
<p align="center">
<img src=figs/celebA_3.png width="300"> <img src=figs/celebA_4.png width="300">
<p align="center">
<img src=figs/celebA_5.png>
</p>
<br>

## Acknowledge

We referred to some evaluation code of [InfoGAN-CR](https://github.com/fjxmlzn/InfoGAN-CR).

We referred to some visualization code of [FactorVAE](https://github.com/1Konny/FactorVAE).

## Citation

