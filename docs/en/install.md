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

1. Installation errors. 
    * If you are installing `cv2` for the first time, `ImportError: libGL.so.1` will occur, which can be solved by `apt install libgl1-mesa-glx`.
    * Errors might occur with `hickle` and this dependency when using KittiCaltech dataset. You can solve the issues by installing additional packages according to the output message.
    * As for WeatherBench, you encounter some import or runtime errors in the version of `xarray`. You can install the latest version or `xarray==0.19.0` to solve the errors, i.e., `pip install xarray==0.19.0`, and then install required packages according to error messages.

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
    │   ├── set06
    │   ├── ...
    │   ├── set10
    │   ├── data_cache.npy
    │   ├── indices_cache.npy
    ├── human
    |   ├── images
    |   ├── test.txt
    |   ├── train.txt
    |── kitti_hkl
    |   ├── sources_test_mini.hkl
    |   ├── ...
    |   ├── X_train.hkl
    │   ├── X_val.hkl
    |── kth
    |   ├── boxing
    |   ├── ...
    |   ├── walking
    |── moving_fmnist
    |   ├── fmnist_test_seq.npy
    |   ├── train-images-idx3-ubyte.gz
    |── moving_mnist
    |   ├── mnist_test_seq.npy
    |   ├── train-images-idx3-ubyte.gz
    |── taxibj
    |   ├── dataset.npz
    |── mv_weather_1_40625deg
    |   ├── q
    |   |   ├── test
    |   |   ├── train
    |   |   ├── val
    |   |   ├── mean.pkl
    |   |   ├── std.pkl
    |   ├── t
    |   ├── ...
    |   ├── z
    |── weather  # single-variant
    |   ├── 2m_temperature
    |   ├── ...
    |── weather_1_40625deg  # single-variant
    |   ├── 2m_temperature
    |   ├── ...
```

### Moving MNIST / FMNIST

[Moving MNIST](http://arxiv.org/abs/1502.04681) and [Moving FMNIST](http://arxiv.org/abs/1502.04681) are toy datasets, which generate gray-scale videos (64x64 resolutions) with two objects. We provide [download_mmnist.sh](https://github.com/Westlake-AI/UniWeat/tree/main/tools/prepare_data/download_mmnist.sh) and [download_mfmnist.sh](https://github.com/Westlake-AI/UniWeat/tree/main/tools/prepare_data/download_mfmnist.sh), which download datasets from [MMNIST download](http://www.cs.toronto.edu/~nitish/unsupervised_video/) and [MFMNIST download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz). Note that the train set is generated online while the test set is fixed to ensure the consistency of evaluation results.

### KittiCaltech Pedestrian

The KittiCaltech Pedestrian dataset uses [Kitti Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) as the train set (2042 videos) and uses [Caltech Pedestrian](https://data.caltech.edu/records/f6rph-90m20) as the test set (1983 videos). We provide [download_kitticaltech.sh](https://github.com/Westlake-AI/UniWeat/tree/main/tools/prepare_data/download_kitticaltech.sh) to prepare the datasets. The data preprocessing of RGB videos (128x160 resolutions) and experiment settings are adopted from [PredNet](https://github.com/coxlab/prednet).

### KTH Action

The [KTH Action](https://ieeexplore.ieee.org/document/1334462) dataset contains grey-scale videos (128x128 resolutions) of six types of human actions performed several times by 25 subjects in four different scenarios. It has 5200 and 3167 videos for the train and test sets and can be downloaded from [KTH download](https://www.csc.kth.se/cvap/actions/). We provide [download_kth.sh](https://github.com/Westlake-AI/UniWeat/tree/main/tools/prepare_data/download_kth.sh) to prepare the dataset. The data preprocessing and experiment settings are adopted from [KTH](https://ieeexplore.ieee.org/document/1334462) and [PredRNN](https://github.com/thuml/predrnn-pytorch).

### Human 3.6M

The [Human3.6M](http://vision.imar.ro/human3.6m/pami-h36m.pdf) dataset contains high-resolution videos (1024x1024 resolutions) of seventeen scenarios of human actions performed by eleven professional actors, which can be downloaded from [Human3.6M download](http://vision.imar.ro/human3.6m/description.php). We provide [download_human3.6m.sh](https://github.com/Westlake-AI/UniWeat/tree/main/tools/prepare_data/download_human3.6m.sh) to prepare the dataset. We borrow the train and test splitting files from [STRPM](https://github.com/ZhengChang467/STRPM) but use 256x256 resolutions in our experiments.

### Multi-variant WeatherBench

[WeatherBench](https://arxiv.org/abs/2002.00469) is the publicly available dataset for global weather prediction, which can be downloaded and processed from [WeatherBench download](https://github.com/pangeo-data/WeatherBench). In the multi-variant version, we choose the important weather variants ('z', 'q', 't', 'u', 'v', 't2m', 'u10', and 'v10') with various vertical levels at high resolutions (128x256). You can download the specific dataset of WeatherBench with [download_weatherbench.sh](https://github.com/Westlake-AI/UniWeat/tree/main/tools/prepare_data/download_weatherbench.sh). Note that `5.625deg` and `1.40625deg` indicate 32x64 and 128x256 resolutions, and the data can have multiple channels. The dataloader is defined in [dataloader_weather_bench_mv](https://github.com/Westlake-AI/UniWeat/tree/main/uniweat/datasets/dataloader_weather_bench_mv.py).

### WeatherBench

[WeatherBench](https://arxiv.org/abs/2002.00469) is the publicly available dataset for global weather prediction, which can be downloaded and processed from [WeatherBench download](https://github.com/pangeo-data/WeatherBench). We choose some important weather variants with certain vertical levels and resolutions, e.g., 2m_temperature, relative_humidity, and total_cloud_cover. You can download the specific dataset of WeatherBench with [download_weatherbench.sh](https://github.com/Westlake-AI/UniWeat/tree/main/tools/prepare_data/download_weatherbench.sh). Note that `5.625deg` and `1.40625deg` indicate 32x64 and 128x256 resolutions, and the data can have multiple channels. The single-variant dataloader is defined in [dataloader_weather_bench](https://github.com/Westlake-AI/UniWeat/tree/main/uniweat/datasets/dataloader_weather_bench.py).

### TaxiBJ

[TaxiBJ](https://arxiv.org/abs/1610.00081) is a popular traffic trajectory prediction dataset, which contains the trajectory data (32x32) in Beijing collected from taxicab GPS with two channels, which can be downloaded from [OneDrive](https://1drv.ms/f/s!Akh6N7xv3uVmhOhDKwx3bm5zpHkDOQ) or [Baidu Cloud](http://pan.baidu.com/s/1qYq7ja8). We borrow the data preprocessing scripts from [DeepST](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ) and provide the processed data in our [Baidu Cloud](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk).
