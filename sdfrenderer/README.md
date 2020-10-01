## SDF-Renderer

Official [PyTorch](https://pytorch.org/) implementation of the SDF-Renderer used in CVPR 2020 paper "Autolabeling 3D Objects With Differentiable Rendering of SDF Shape Priors" by the ML Team at [Toyota Research Institute (TRI)](https://www.tri.global/), cf. [References](#references) below.
[**[Full paper]**](https://arxiv.org/pdf/1911.11288.pdf) [**[YouTube]**](https://www.youtube.com/watch?v=Utzj-kfWHP4)

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="/sdfrenderer/media/figs/tri-logo.png" width="20%"/>
</a>

<a href="https://www.youtube.com/watch?v=Utzj-kfWHP4" target="_blank">
<img width="60%" src="/sdfrenderer/media/figs/sdfrenderer-teaser.gif"/>
</a>


## Running the Code
Download the [data folder](https://drive.google.com/file/d/1cvLeXDhaghjzCK-gmnQbxh8rTvd71miw/view?usp=sharing) containing the pretrained DeepSDF network.
To test the renderer using the pretrained DeepSDF network, run the following code:
```
python main.py --model ../data/nets/deepsdf.pt
```
Alternatively you can use just the surfel part of the renderer to render 3D models. 
To test it, specify the path to the provided car model from the [HomebrewedDB dataset](http://campar.in.tum.de/personal/ilic/homebreweddb/index.html).
```
python main.py --model ../data/db/car.ply 
```

## Zero-Isosurface Projection

<a href="https://www.youtube.com/watch?v=Utzj-kfWHP4" target="_blank">
<img width="60%" src="/sdfrenderer/media/figs/projection.gif"/>
</a>

To obtain surface points we apply zero-isosurface projection on the grid points using regressed by DeepSDF SDF values.
```
# Get DeepSDF output
pred_sdf_grid = dsdf(inputs)

# Get surface points using 0-isosurface projection
points, nocs, normals = grid_3d.get_surface_points(pred_sdf_grid)
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

The repository uses parts of the <a href="https://github.com/facebookresearch/DeepSDF" target="_blank"> DeepSDF project </a>