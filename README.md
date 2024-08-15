![plantcamo](https://github.com/yjybuaa/PlantCamo/assets/39208339/9b6888db-cd9d-46f0-b851-d40726788cf4)

<div align=center>
<a src="https://img.shields.io/badge/%F0%9F%93%96-Arxiv_xxxx.xxxxx-red.svg?style=flat-square" href="https://arxiv.org/abs/xxxx.xxxxx">
<img src="https://img.shields.io/badge/%F0%9F%93%96-Arxiv_xxxx.xxxxx-red.svg?style=flat-square">
</a>
<a src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-ed6c00.svg?style=flat-square" href="https://zhengfenglab.com/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-ed6c00.svg?style=flat-square">
</a>
</div>

__PlantCamo Dataset__ is the first dataset dedicated for plant camouflage detection. It contains over 1,000 images with plant camouflage characteristics.

## Demo
https://github.com/yjybuaa/PlantCamo/assets/39208339/766645f1-6951-4cf0-baf1-8a5fb4b13bd9



## Download the Dataset and Results
[PlantCamo-full](https://pan.baidu.com/s/1sLFavm2LjVaEo_5uAlsIDQ)(Code: nm46)  [Google drive link](https://drive.google.com/file/d/1-HsLXjzR-27VyARum071h4JF_RprOKu_/view?usp=sharing)

[PlantCamo-Train-and-Test](https://pan.baidu.com/s/1vdR-kj63qvsT3M4-wkgMoQ)(Code: hq87)   [Google drive link](https://drive.google.com/file/d/1eMvSbNJJbh6BYea-3ZktzDReHFujsb1_/view?usp=drive_link)

[Results](https://pan.baidu.com/s/14W4oH2UX2MlRJ2H5ewE1yA)(Code: 6o76)  [Google drive link](https://drive.google.com/file/d/1-7kaOlZJRSDYrIdyWAjJqXCEGS9QX2_T/view?usp=drive_link)

>## Usage
>
>>The training and testing experiments are conducted using PyTorch with a single RTX 3090 GPU of 24 GB Memory.
>>
>>Download `pvt_v2_b2.pth` at [here](https://pan.baidu.com/s/11dSkyGKb71lT_7HxSCiIjw) (Code: gy87) or [Google drive link](https://drive.google.com/file/d/1-AFs2dP3p0OMw3Vbnf0tMsgkYbpT6C23/view?usp=sharing), and put it into `.\pretrained_pvt`

## Train

Download `PlantCamo-Train-and-Test` at [here](https://pan.baidu.com/s/1vdR-kj63qvsT3M4-wkgMoQ)(Code: hq87) or [Google drive link](https://drive.google.com/file/d/1eMvSbNJJbh6BYea-3ZktzDReHFujsb1_/view?usp=drive_link), and put it into `.\datasets`

## Test

Download trained model `Net_epoch_best.pth` at [here](https://pan.baidu.com/s/1eZpqxx5b5Y_V9klLQxf5WQ)(Code: b98f) or [Google drive link](https://drive.google.com/file/d/1-GP18g79xszyAnvZ1ZzTXzhFoH7gzYed/view?usp=sharing), and put it into `.\ckpt`

## Evaluation

You can find it in [https://github.com/lartpang/PySODMetrics](https://github.com/lartpang/PySODMetrics) or you can run the `metric_caller.py`

## Citation
We appreciate your support of our work!
```bibtex
