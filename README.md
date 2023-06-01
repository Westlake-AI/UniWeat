# UniWeat: Unified Toolbox for Weather Prediction

[📘Documentation](https://openstl.readthedocs.io/en/latest/) |
[🛠️Installation](docs/en/install.md) |
[🆕News](docs/en/changelog.md)

This repository is an open-source project for weather prediction (single and multiple variable prediction), which is built as an extensive project of [OpenSTL](https://github.com/chengtan9907/OpenSTL). We are working on it and new features is updating!

## News and Updates

[2023-04-27] `UniWeat` v0.1.0 is initalized (on updating).

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/Westlake-AI/UniWeat
cd UniWeat
conda env create -f environment.yml
conda activate UniWeat
python setup.py develop  # or `pip install -e .`
```

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

Please refer to [install.md](docs/en/install.md) for more detailed instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage. Here is an example of single GPU non-distributed training SimVP on Moving MNIST dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview of Methods and Datasets

We support various spatiotemporal prediction learning (STL) methods and will provide benchmarks on various STL datasets.

* Spatiotemporal Prediction Methods.

    <details open>
    <summary>Currently supported methods</summary>

    - [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NIPS'2015)
    - [x] [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) (NIPS'2017)
    - [x] [PredRNN++](https://arxiv.org/abs/1804.06300) (ICML'2018)
    - [x] [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2018)
    - [x] [MAU](https://arxiv.org/abs/1811.07490) (CVPR'2019)
    - [x] [CrevNet](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2020)
    - [x] [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
    - [x] [PredRNN.V2](https://arxiv.org/abs/2103.09504v4) (TPAMI'2022)
    - [x] [SimVP](https://arxiv.org/abs/2206.05099) (CVPR'2022)
    - [x] [SimVP.V2](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
    - [x] [TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
    - [x] [DMVFN](https://arxiv.org/abs/2303.09875) (CVPR'2023)

    </details>

    <details open>
    <summary>Currently supported MetaFormer models for SimVP</summary>

    - [x] [ViT](https://arxiv.org/abs/2010.11929) (ICLR'2021)
    - [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030) (ICCV'2021)
    - [x] [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NIPS'2021)
    - [x] [ConvMixer](https://arxiv.org/abs/2201.09792) (Openreview'2021)
    - [x] [UniFormer](https://arxiv.org/abs/2201.09450) (ICLR'2022)
    - [x] [PoolFormer](https://arxiv.org/abs/2111.11418) (CVPR'2022)
    - [x] [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR'2022)
    - [x] [VAN](https://arxiv.org/abs/2202.09741) (ArXiv'2022)
    - [x] [IncepU (SimVP.V1)](https://arxiv.org/abs/2206.05099) (CVPR'2022)
    - [x] [gSTA (SimVP.V2)](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
    - [x] [HorNet](https://arxiv.org/abs/2207.14284) (NIPS'2022)
    - [x] [MogaNet](https://arxiv.org/abs/2211.03295) (ArXiv'2022)

    </details>

* Spatiotemporal Predictive Learning Benchmarks.

    <details open>
    <summary>Currently supported datasets</summary>

    - [x] [Human3.6M](http://vision.imar.ro/human3.6m/pami-h36m.pdf) (TPAMI'2014)  [[download](http://vision.imar.ro/human3.6m/description.php)] [[config](configs/human)]
    - [x] [KTH Action](https://ieeexplore.ieee.org/document/1334462) (ICPR'2004)  [[download](https://www.csc.kth.se/cvap/actions/)] [[config](configs/kth)]
    - [x] [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) (IJRR'2013) [[download](https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip)] [[config](configs/kitticaltech)]
    - [x] [Moving MNIST](http://arxiv.org/abs/1502.04681) (ICML'2015) [[download](http://www.cs.toronto.edu/~nitish/unsupervised_video/)] [[config](configs/mmnist)]
    - [x] [Moving FMNIST](http://arxiv.org/abs/1502.04681) (ICML'2015) [[download](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk)] [[config](configs/mfmnist)]
    - [x] [TaxiBJ](https://arxiv.org/abs/1610.00081) (AAAI'2017) [[download](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)] [[config](configs/taxibj)]
    - [x] [WeatherBench](https://arxiv.org/abs/2002.00469) (ArXiv'2020) [[download](https://github.com/pangeo-data/WeatherBench)] [[config](configs/weather)]

    </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

UniWeat is an open-source project for single and multiple variable weather prediction applications created by researchers in **CAIRI AI Lab**. We encourage researchers interested in weather prediction to contribute to UniWeat! UniWeat is an extensive project of [OpenSTL](https://github.com/chengtan9907/OpenSTL).

## Citation

If you are interested in our repository or our paper, please cite the following paper:

```
@misc{li2023uniweat,
  title={UniWeat: Unified Toolbox for Weather Prediction},
  author={Li, Siyuan and Lin, Haitao and Tan, Cheng and Chen, Lei and Li, Stan Z},
  howpublished = {\url{https://github.com/Westlake-AI/UniWeat}},
  year={2023}
}
```

## Contribution and Contact

For adding new features, needing helps, or reporting bugs associated with `UniWeat`, please open a [GitHub issue](https://github.com/Westlake-AI/UniWeat/issues) and [pull request](https://github.com/Westlake-AI/UniWeat/pulls) with the tag "new features", "help wanted", or "enhancement". Feel free to contact us through email if you have any questions. Enjoy!

- Siyuan Li (lisiyuan@westlake.edu.cn), Westlake University & Zhejiang University
- Haitao Lin (linhaitao@westlake.edu.cn), Westlake University & Zhejiang University
- Cheng Tan (tancheng@westlake.edu.cn), Westlake University & Zhejiang University

<p align="right">(<a href="#top">back to top</a>)</p>
