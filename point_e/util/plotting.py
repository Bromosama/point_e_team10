from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from .point_cloud import PointCloud
import seaborn as sns
#sns.set_style(rc = {'axes.facecolor': 'lightsteelblue'})
sns.set_style("darkgrid")
sns.set_style("darkgrid", {'grid.color': 'lightsteelblue', 'axes.facecolor': 'white'})

def plot_point_cloud(
    pc: PointCloud,
    color: bool = True,
    grid_size: int = 1,
    fixed_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = (
        (-0.75, -0.75, -0.75),
        (0.75, 0.75, 0.75),
    ),
):
    """
    Render a point cloud as a plot to the given image path.

    :param pc: the PointCloud to plot.
    :param image_path: the path to save the image, with a file extension.
    :param color: if True, show the RGB colors from the point cloud.
    :param grid_size: the number of random rotations to render.
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(grid_size, grid_size, 1 + j + i * grid_size, projection="3d")
            color_args = {}
            if color:
                color_args["c"] = np.stack(
                    [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
                )
            c = pc.coords

            if grid_size > 1:
                theta = np.pi * 2 * (i * grid_size + j) / (grid_size**2)
                rotation = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                c = c @ rotation

            ax.scatter(c[:, 0], c[:, 1], c[:, 2], **color_args, s = 5)

            if fixed_bounds is None:
                min_point = c.min(0)
                max_point = c.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                ax.set_xlim3d(center[0] - size, center[0] + size)
                ax.set_ylim3d(center[1] - size, center[1] + size)
                ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])

    return fig


def plot_attention_cloud(
    pc: PointCloud,
    color: bool = False,
    grid_size: int = 1,
    fixed_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = (
        (-0.75, -0.75, -0.75),
        (0.75, 0.75, 0.75),
    ),
    col: float = np.ones(1024), 
    track_idx: int = None,
    alpha_val: float = 1 
):
    """
    Render a point cloud as a plot to the given image path.

    :param pc: the PointCloud to plot.
    :param image_path: the path to save the image, with a file extension.
    :param color: if True, show the RGB colors from the point cloud.
    :param grid_size: the number of random rotations to render.
    """
    fig = plt.figure(figsize=(8, 8))
    axes = []

    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(grid_size, grid_size, 1 + j + i * grid_size, projection="3d")
            axes.append(ax)
            color_args = {}
            if color:
                color_args["c"] = np.stack(
                    [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
                )
            
            c = pc.coords

            if grid_size > 1:
                theta = np.pi * 2 * (i * grid_size + j) / (grid_size**2)
                rotation = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                c = c @ rotation
            if color:
                im = ax.scatter(c[:, 0], c[:, 1], c[:, 2], **color_args)
            else:
                im = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c = col, cmap = 'rocket_r',  alpha = alpha_val) # vmin=0, vmax=1,
                if track_idx != None:
                    im = ax.scatter(c[track_idx, 0], c[track_idx, 1], c[track_idx, 2], c = [1], cmap = 'rocket_r', vmin=0, vmax=1)
                
            if fixed_bounds is None:
                min_point = c.min(0)
                max_point = c.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                ax.set_xlim3d(center[0] - size, center[0] + size)
                ax.set_ylim3d(center[1] - size, center[1] + size)
                ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])
            
            if i == 2:
                c = fig.colorbar(im, ax = ax, location = 'bottom')
            
    #axes = np.array(axes)
    #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], 
                              #shrink=0.5, location = 'bottom')
    
    #kw['orientation'] = 'horizontal'
    #fig.colorbar(im, cax=cax, **kw)



    #fig.colorbar(im, ax = axes.ravel().tolist())

    #fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')
    
    return fig


def plot_attention_index(
    pc: PointCloud,
    color: bool = False,
    grid_size: int = 1,
    fixed_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = (
        (-0.75, -0.75, -0.75),
        (0.75, 0.75, 0.75),
    ),
    col: float = np.ones(1024), 
    track_idx: int = None,
    alpha_val: float = 1 
):
    """
    Render a point cloud as a plot to the given image path.

    :param pc: the PointCloud to plot.
    :param image_path: the path to save the image, with a file extension.
    :param color: if True, show the RGB colors from the point cloud.
    :param grid_size: the number of random rotations to render.
    """
    
    fig = plt.figure(figsize=(8, 8))
    axes = []
    N = 1024

    x = np.linspace(0, N-1, N)
    y = np.linspace(0, N-1, N)

    X, Y = np.meshgrid(x, y)
    grid = np.dstack((X, Y, col))
    grid = grid.reshape(-1, grid.shape[-1])
    
    
    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(grid_size, grid_size, 1 + j + i * grid_size, projection="3d")
            axes.append(ax)
            color_args = {}
            if color:
                color_args["c"] = np.stack(
                    [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
                )
            
            c = grid
            if grid_size > 1:
                theta = np.pi * 2 * (i * grid_size + j) / (grid_size**2)
                rotation = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                c = c @ rotation
            if color:
                im = ax.scatter(c[:, 0], c[:, 1], c[:, 2], **color_args)
            else:
                im = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c = col, s = 2, cmap = 'rocket_r',  alpha = alpha_val, vmin=0, vmax=1)
                if track_idx != None:
                    im = ax.scatter(c[track_idx, 0], c[track_idx, 1], c[track_idx, 2], c = [1], cmap = 'rocket_r', vmin=0, vmax=1)
                
            fixed_bounds = None
            if fixed_bounds is None:
                min_point = c.min(0)
                max_point = c.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                #ax.set_xlim3d(center[0] - size, center[0] + size)
                #ax.set_ylim3d(center[1] - size, center[1] + size)
                #ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])
            
            if i == 2:
                c = fig.colorbar(im, ax = ax, location = 'bottom')
            
    #axes = np.array(axes)
    #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], 
                              #shrink=0.5, location = 'bottom')
    
    #kw['orientation'] = 'horizontal'
    #fig.colorbar(im, cax=cax, **kw)



    #fig.colorbar(im, ax = axes.ravel().tolist())

    #fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')
    
    return fig