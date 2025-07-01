
# RAISE: Refine Any Object in Any Scene
[![arXiv](https://img.shields.io/badge/arXiv-2506.23835-b31b1b.svg)](https://arxiv.org/abs/2506.23835)

![Preview](https://github.com/tmKamal/hosted-images/blob/master/under-construction/Document.gif?raw=true)

## Environment


To set up the environment, you can use the provided `environment.yml` file. This file contains all the necessary dependencies for running the code.

Otherwise, you can manually install the required packages according to `setup.bash`.

Especially for Trellis environment, you can follow the instructions in the [Trellis repository](https://github.com/microsoft/TRELLIS).

## Folder Structure

```
objects_office_d455/
├── depths_cam/             # Contains camera-based depth data. Optional.
├── depths_est/             # Contains estimated depth data. Optional.
├── images/                 # Contains image files.
├── sparse/                 # Contains sparse reconstruction data in COLMAP format. (W2C)
├── description.yml         # YAML file describing dataset details.
└── split.yml               # YAML file specifying dataset splits.
```

## Pipeline

### 2D Segmentation

``` bash
python ./segmentation_2d.py -s ${data_path}
```

After completing this process, the masks for each target object will be located in the directory named `masked_image_rgba` within the source path.

### Reconstruction

``` bash
python train_3dgs.py \
    -s ${data_path} \
    -m ${model_path} \
    --test_iterations -1 \
    --eval \ 
    --split_yml_name ${split_yml_name} \
```

### 3D Segmentation

``` bash
python segmentation_3dgs.py \
    -s ${data_path} \
    -m ${model_path} \
    -r 1 \
    --eval \
    --split_yml_name ${split_yml_name}
```

### View Selection

``` bash
python view_selection.py \
    -s ${data_path} \
    -m ${model_path} \
    -r 1 \
    --eval \
    --split_yml_name ${split_yml_name}
```

### Generation

``` bash
${trellis_env}/bin/python trellis_img2gs.py -m ${model_path} 
```

### Truncation

``` bash
python truncation_opacity.py \
    -m ${model_path} \
    --threshold 0.1 \
```

### Alignment

``` bash
python align_3dgs_clpe_9dof.py \
    -s ${data_path} \
    -m ${model_path} \
    --eval \
    --split_yml_name ${split_yml_name}
```

### Refinement

``` bash
python post_refine_gs.py \
    -s ${data_path} \
    -m ${model_path} \
    -r 1 \
    --images ${data_path}/masked_image_rgba/masked \
    --sh_degree 0 \
    --iterations 800 \
    --eval \
    --split_yml_name ${split_yml_name}
```

## Acknowledgements

We extend our sincere gratitude to the following tools and libraries that were instrumental in the successful completion of this project:

- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting): 3DGS scene reconstruction.
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting): 2DGS scene reconstruction.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO): To find object locations in images.
- [SAM 2](https://github.com/facebookresearch/segment-anything-2): To track masks throughout the frames.
- [MASt3R](https://github.com/naver/mast3r): To finish the feature extraction and matching between images.
- [Trellis](https://github.com/microsoft/TRELLIS): To generate 3D objects represented by Gaussians from images.

## Citation
If you find this paper or the code helpful for your work, please consider citing our preprint,
```
@misc{chen2025refineobjectscene,
      title={Refine Any Object in Any Scene}, 
      author={Ziwei Chen and Ziling Liu and Zitong Huang and Mingqi Gao and Feng Zheng},
      year={2025},
      eprint={2506.23835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23835}, 
}
```