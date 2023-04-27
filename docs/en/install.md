# Installation

## Install the project

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/Westlake-AI/UniWeat
cd UniWeat
conda env create -f environment.yml
conda activate UniWeat
python setup.py develop  # or `pip install -e .`
```

<details close>
<summary>Requirements</summary>

* Linux (Windows is not officially supported)
* Python 3.7+
* PyTorch 1.8 or higher
* CUDA 10.1 or higher
* NCCL 2
* GCC 4.9 or higher
</details>

<details close>
<summary>Dependencies</summary>

* argparse
* fvcore
* numpy
* hickle
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
</details>

**Note:**

1. Some errors might occur with `hickle` and `xarray` when using KittiCaltech and WeatherBench datasets. As for KittiCaltech, you can solve the issues by installing additional pacakges according to the output messeage. As for WeatherBench, you can install the latest version of `xarray` to solve the errors, i.e., `pip install xarray==0.19.0` and then installing required pacakges according to error messages.

2. Following the above instructions, OpenSTL is installed on `dev` mode, any local modifications made to the code will take effect. You can install it by `pip install .` to use it as a PyPi package, and you should reinstall it to make the local modifications effect.

## Prepare datasets

It is recommended to symlink your dataset root (assuming `$YOUR_DATA_ROOT`) to `$UNIWEAT/data`. If your folder structure is different, you need to change the corresponding paths in config files.

We support following datasets: [Human3.6M](http://vision.imar.ro/human3.6m/pami-h36m.pdf) [[download](http://vision.imar.ro/human3.6m/description.php)], [KTH Action](https://ieeexplore.ieee.org/document/1334462) [[download](https://www.csc.kth.se/cvap/actions/)], [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) [[download](https://figshare.com/articles/dataset/KITTI_hkl_files/7985684)], [Moving MNIST](http://arxiv.org/abs/1502.04681) [[download](http://www.cs.toronto.edu/~nitish/unsupervised_video/)], [TaxiBJ](https://arxiv.org/abs/1610.00081) [[download](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)], [WeatherBench](https://arxiv.org/abs/2002.00469) [[download](https://github.com/pangeo-data/WeatherBench)]. Please prepare datasets with tools and scripts under `tools/prepare_data`.

You can also download the version we used in experiments from [**Baidu Cloud**](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk) (kjfk). Please do not distribute the datasets and only use them for research.

```
UniWeat
├── configs
└── data
    ├── caltech
    ├── human
    |   ├── images
    |   ├── test.txt
    |   ├── train.txt
    |── kitti_hkl
    |   ├── sources_test_mini.hkl
    |   ├── X_train.hkl
    |   ├── ...
    |── kth
    |   ├── boxing
    |   ├── ...
    |── moving_fmnist
    |── moving_mnist
    |   ├── mnist_test_seq.npy
    |   ├── train-images-idx3-ubyte.gz
    |── taxibj
    |   ├── dataset.npz
    |── weather
    |   ├── 2m_temperature
    |   ├── ...
    |── weather_1_40625deg
    |   ├── 2m_temperature
    |   ├── ...
```
