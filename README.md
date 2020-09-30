## Autolabeling 3D Objects With Differentiable Rendering of SDF Shape Priors

Official [PyTorch](https://pytorch.org/) implementation of the CVPR 2020 paper "Autolabeling 3D Objects With Differentiable Rendering of SDF Shape Priors" by the ML Team at [Toyota Research Institute (TRI)](https://www.tri.global/), cf. [References](#references) below.
[**[Full paper]**](https://arxiv.org/pdf/1911.11288.pdf) [**[YouTube]**](https://www.youtube.com/watch?v=Utzj-kfWHP4)

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="/media/figs/tri-logo.png" width="20%"/>
</a>

<a href="https://www.youtube.com/watch?v=Utzj-kfWHP4" target="_blank">
<img width="60%" src="/media/figs/sdflabel-teaser.gif"/>
</a>


## Setting up your environment
To set up the environment using conda, use the following commands:
```
conda env create -n sdflabel -f environment.yml
conda activate sdflabel
```

## Optimization demo
To run the optimization demo first download the data folder using the following [link](https://pytorch.org/).

Then, run the following command:
```
python main.py config/config_train.ini --train
```

## Training CSS network
To train the CSS network run the following command:
```
python main.py configs/config_refine.ini --demo
```

### Dataset format
The dataset of crops represents a collection of detected *RGB patches (CSS input)*, corresponding *NOCS patches (CSS output)*, and a JSON DB file comprising the patch relevant information, most importantly *SDF latent vector* corresponding to the depicted 3D model.
The example of the dataset is located in the *data/crops_db* folder.

## Optimization
To run optimization on the detected frames run the following command:
```
python main.py config/config_refine.ini --refine
```
Please modify the config file to specify the path to KITTI. Upon completion, autolabels will be stored to the *output* folder specified in the config file (*output* -> *labels*).
To evaluate the generated dump, run:
```
python main.py config/config_refine.ini --evaluate
```



## License

The source code is released under the [MIT license](LICENSE.md).

## References

#### Autolabeling 3D Objects With Differentiable Rendering of SDF Shape Priors (CVPR 2020 oral)
*Sergey Zakharov\*, Wadim Kehl\*, Arjun Bhargava, Adrien Gaidon*

```
@inproceedings{sdflabel,
author = {Sergey Zakharov and Wadim Kehl and Arjun Bhargava and Adrien Gaidon},
title = {Autolabeling 3D Objects with Differentiable Rendering of SDF Shape Priors},
booktitle = {IEEE Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
