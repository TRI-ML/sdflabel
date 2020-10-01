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
Add the sdfrenderer directory to *PYTHONPATH*:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/sdfrenderer"
```

## Optimization demo
To run the optimization demo, first download the [data folder](https://drive.google.com/file/d/1cvLeXDhaghjzCK-gmnQbxh8rTvd71miw/view?usp=sharing).
Then, extract the archive to the root folder of the project and run the following command:
```
python main.py configs/config_refine.ini --demo
```

## Training CSS network
To train the CSS network, run the following command:
```
python main.py configs/config_train.ini --train
```

### Dataset format
The dataset of crops represents a collection of detected *RGB patches (CSS input)*, corresponding *NOCS patches (CSS output)*, and a JSON DB file comprising the patch relevant information (most importantly *SDF latent vectors* corresponding to the depicted 3D models).
An example of such dataset is located in the *data/db/crops* folder.

## Optimization
Download [KITTI 3D](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and modify the *kitti_path* in the config file *config_refine.ini* accordingly.
To run optimization on the KITTI 3D dataset, run the following command:
```
python main.py configs/config_refine.ini --refine
```
Upon completion, autolabels will be stored to the *output* folder specified in the config file (*output* -> *labels*).
To evaluate the generated dump, run:
```
python main.py configs/config_refine.ini --evaluate
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
