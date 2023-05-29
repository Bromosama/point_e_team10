# Indoor scene generation using Point-E and Training Free Layout Control

Authors: *Luyang Busser, Alessia Hu, Oline Ranum, Luc Sträter, Sina Taslimi, Miranda Zhou*

This repository contains code and blogpost on the reproduction and extension of [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://arxiv.org/abs/2212.08751), 2022. We present framework for extending point cloud diffusion models to accomodate indoor scene generations directly from text prompts. The framework is able to produce small scenes composed of 2-3 furnitures. For an in-depth discussion of our work, see the paper.

<p align="center">
   <img src="src/imgs/results/Diffusion.gif" width = 500> 
   <br>
   <text><em>Animation of four 3D point clouds diffusing.</em></text>
</p>

The official code and model release for Point-E can be found at [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://github.com/openai/point-e).

The official code and model release for Training Free layout control can be found at [Training-Free Layout Control with Cross-Attention Guidance](https://github.com/silent-chen/layout-guidance).

## Code structure

| Directory | Description |
| --------- | ----------- |
| `demos/` | Notebooks used to analyze training runs, results and render scenes. |
| `src/layout_guidance` | Source code used for Training-Free Layout Control with Cross-Attention Guidance. Adapted from the [repo of the original paper](https://github.com/silent-chen/layout-guidance). |
| `src/point_e` | Source code used for point-e point cloud diffusion. Adapted from the [repo of the original paper](https://github.com/openai/point-e). |
| `src/imgs/` | Location where produced images are stored. |
| `src/scripts/` | Files to run that reproduce the results. |
| `src/train_results/` | Location where all trained models and its (intermediate) results are stored. |
| `paper.pdf` | Report introducing original work, discussing our novel contribution and analaysis. |


# Usage


First, install the conda environment 
```shell
conda create -n isg python=3.8
source activate isg

pip install -r requirements.txt
```

# Usage 

To get started with examples, see the following notebooks:

 * [image2pointcloud.ipynb](src/point_e/examples/image2pointcloud.ipynb) - sample a point cloud, conditioned on some example synthetic view images.
 * [text2pointcloud.ipynb](src/point_e/examples/text2pointcloud.ipynb) - use our small, worse quality pure text-to-3D model to produce 3D point clouds directly from text descriptions. This model's capabilities are limited, but it does understand some simple categories and colors.
 * [pointcloud2mesh.ipynb](src/point_e/examples/pointcloud2mesh.ipynb) - try our SDF regression model for producing meshes from point clouds.

For our P-FID and P-IS evaluation scripts, see:

 * [evaluate_pfid.py](src/point_e/evals/scripts/evaluate_pfid.py)
 * [evaluate_pis.py](src/point_e/evals/scripts/evaluate_pis.py)

For our Blender rendering code, see [blender_script.py](src/point_e/evals/scripts/blender_script.py)

# Samples

You can download the seed images and point clouds corresponding to the paper banner images [here](https://openaipublic.azureedge.net/main/point-e/banner_pcs.zip).

You can download the seed images used for COCO CLIP R-Precision evaluations [here](https://openaipublic.azureedge.net/main/point-e/coco_images.zip).
