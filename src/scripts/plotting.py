from typing import Optional, Tuple

# Standard imports
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import matplotlib.ticker as ticker
import sys, os 

# Detect local paths
src_path = os.path.join(sys.path[0], '../')
sys.path.insert(1, src_path)

#sns.set_style(rc = {'axes.facecolor': 'lightsteelblue'})
#sns.set_style("darkgrid")
sns.set_style("white")
#sns.set_style("darkgrid", {'grid.color': 'lightsteelblue', 'axes.facecolor': 'white'})

from src.point_e.util.point_cloud import PointCloud


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
    formatter = ticker.ScalarFormatter(useMathText=True)

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
                im = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c = col, s = 5, cmap = 'rocket_r',  alpha = alpha_val) # vmin=0, vmax=1,
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

            decimals = 2  # Set the desired number of decimals
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 1))
            
            ax.zaxis.set_major_formatter(ticker.FormatStrFormatter(f'%.{decimals}f'))
            num_ticks = 3  # Set the desired number of ticks
            ticks = ticker.MaxNLocator(num_ticks)
            ax.yaxis.set_major_locator(ticks)
            ax.xaxis.set_major_locator(ticks)
            ax.set_zticks([])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.grid(False)
            ax.w_xaxis.line.set_color('none')
            ax.w_yaxis.line.set_color('none')
            ax.w_zaxis.line.set_color('none')
            
            ax.set_facecolor('white')  # Set the background color to white
            ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))  # Set the axis pane color to white (optional)
            ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))  # Set the axis pane color to white (optional)
            ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))  # Set the axis pane color to white (optional)    
            
            
            if i == 2:
                cbar = fig.colorbar(im, ax = ax, location = 'bottom')
                num_ticks = 2  # Set the desired number of ticks
                ticks = ticker.MaxNLocator(num_ticks)
                cbar.formatter = formatter
                cbar.locator = ticks
                cbar.ax.tick_params(labelsize=15)
                cbar.update_ticks()
                            # Create the colorbar

            

        # Set the tick formatter to scientific notation

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
    col: float = np.zeros(1024), 
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
    plt.tight_layout()
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
            ax.ticklabel_format(style='sci', axis='y')  #y-axis scientific notation turned off
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
                #ax.ticklabel_format(style='sci', axis='y')  #y-axis scientific notation turned off

            else:
                im = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c = col, s = 2, cmap = 'rocket_r',  alpha = alpha_val)
                ax.ticklabel_format(style='sci', axis='z')  #y-axis scientific notation turned off
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 1))
                cbar.formatter = formatter
                num_ticks = 3  # Set the desired number of ticks
                ticks = ticker.MaxNLocator(num_ticks)
                ax.yaxis.set_major_locator(ticks)
                ax.xaxis.set_major_locator(ticks)
                cbar.locator = ticks
                cbar.ax.tick_params(labelsize=15)
                cbar.update_ticks()
                ax.set_zticks([])
                ax.set_yticks([0, 256,  1281])
                ax.set_xticks([0, 256,  1281])
                # Set the tick size directly in the plot
                plt.tick_params(axis='both', which='major', labelsize=17)
                plt.tick_params(axis='both', which='minor', labelsize=17)
                
                ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
                if track_idx != None:
                    im = ax.scatter(c[track_idx, 0], c[track_idx, 1], c[track_idx, 2], c = [1], cmap = 'rocket_r')#, vmin=0, vmax=1)
                
            fixed_bounds = None
            if fixed_bounds is None:
                min_point = c.min(0)
                max_point = c.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                ax.set_zticks([])
                #ax.set_xlim3d(center[0] - size, center[0] + size)
                #ax.set_ylim3d(center[1] - size, center[1] + size)
                #ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])
            
            if i == 2:
                cbar = fig.colorbar(im, ax = ax, location = 'bottom')
                max_decimals = 2
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 1))
                cbar.formatter = formatter
                
                cbar.update_ticks()
            
    #axes = np.array(axes)
    #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], 
                              #shrink=0.5, location = 'bottom')
    
    #kw['orientation'] = 'horizontal'
    #fig.colorbar(im, cax=cax, **kw)



    #fig.colorbar(im, ax = axes.ravel().tolist())

    #fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')
    
    return fig


def plot_heatmap(
    pc: PointCloud,
    color: bool = False,
    grid_size: int = 1,
    fixed_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = (
        (-0.75, -0.75, -0.75),
        (0.75, 0.75, 0.75),
    ),
    col: float = np.zeros(1024), 
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
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()
    axes = []
    N = col.shape[0]

    x = np.linspace(0, N-1, N)
    y = np.linspace(0, N-1, N)

    X, Y = np.meshgrid(x, y)
    
    grid = np.dstack((X, Y, col))
    grid = grid.reshape(-1, grid.shape[-1])

    

    c = grid
    
    im = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c = col, s = 2, cmap = 'rocket_r',  alpha = alpha_val)
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
    fixed_bounds = None
    if fixed_bounds is None:
        min_point = c.min(0)
        max_point = c.max(0)
        size = (max_point - min_point).max() / 2
        center = (min_point + max_point) / 2
                #ax.set_xlim3d(center[0] - size, center[0] + size)
                #ax.set_ylim3d(center[1] - size, center[1] + size)
                #ax.set_zlim3d(center[2] - size, center[2] + size)

    formatter = ticker.ScalarFormatter(useMathText=True)
    # Remove the background grid
    
    ax.grid(False)
    ax.w_xaxis.line.set_color('none')
    ax.w_yaxis.line.set_color('none')
    ax.w_zaxis.line.set_color('none')
    
    ax.set_facecolor('white')  # Set the background color to white
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))  # Set the axis pane color to white (optional)
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))  # Set the axis pane color to white (optional)
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))  # Set the axis pane color to white (optional)
    # Set the maximum number of decimals on the z-axis
    formatter.set_useOffset(False)
  
    cbar = fig.colorbar(im, ax = ax, shrink = 0.5, location = 'bottom', pad = 0.08)
    
    #cbar.set_ticks(cbar.get_ticks())  # Update the colorbar ticks
    
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 1))
    cbar.formatter = formatter
    num_ticks = 3  # Set the desired number of ticks
    ticks = ticker.MaxNLocator(num_ticks)
    ax.yaxis.set_major_locator(ticks)
    ax.xaxis.set_major_locator(ticks)
    cbar.locator = ticks
    cbar.ax.tick_params(labelsize=15)
    cbar.update_ticks()
    ax.set_zticks([])
    ax.set_yticks([0, 256,  1281])
    ax.set_xticks([0, 256,  1281])
    # Set the tick size directly in the plot
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    



    #axes = np.array(axes)
    #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], 
                              #shrink=0.5, location = 'bottom')
    
    #kw['orientation'] = 'horizontal'
    #fig.colorbar(im, cax=cax, **kw)



    #fig.colorbar(im, ax = axes.ravel().tolist())

    #fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')
    
    return fig

