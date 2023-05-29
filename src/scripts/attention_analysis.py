# Standard imports
import seaborn as sns
sns.set_style("darkgrid")
sns.set_style("darkgrid", {'grid.color': 'lightsteelblue', 'axes.facecolor': 'white'})
import matplotlib.pyplot as plt
import imageio, importlib
from PIL import Image

import nopdb
import math, torch
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm

import sys, os 

# Detect local paths
src_path = os.path.join(sys.path[0], '../')
sys.path.insert(1, src_path)


# Import modules from point_e
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.plotting import plot_attention_cloud
from point_e.util.plotting import plot_attention_index


class AttentionTools():
    """Class used to visualize attention maps

    Args:
        base_model : Model to vizualize.
        sampler: .
       
    """
    def __init__(self, base_model, sampler, breakpoints = None, cut = -1):
        self.base_model = base_model
        self.sampler = sampler

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.parent_dir = os.path.join(sys.path[0], '../')
        self.output_dir = os.path.join(self.parent_dir, 'src/imgs/results/')

        if breakpoints == None:
            self.breakpoints = [0, 10, 20, 30, 40, 50, 60, 64, 120]
        else: 
            self.breakpoints = breakpoints
        
        self.breakpoints_downsample = self.breakpoints[:cut]
        self.time = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.time_downnsample = self.time[:-1]
    
    def make_directory(self, dir):
        path = os.path.join(self.output_dir, dir)
        try: 
            os.mkdir(path)
        except:
            pass
        return path
        

    def est_attention(self, attn_call):
        """ Function to extract all attention components from 
            retrieved map
        """
        attention_scores = attn_call.locals['weight'][0]
        attention_probs = attention_scores.softmax(dim=-1)
        # Average across all heads
        avg_attn = torch.mean(attention_probs, dim = 0)
        
        # Est. self attention
        pc_self_attn = avg_attn[257:, 257:]
        
        # Est. cross attention
        pc_cross_attn = avg_attn[:257, 257:]

        pc_self_attn = pc_self_attn.cpu()
        pc_cross_attn = pc_cross_attn.cpu()
        avg_attn = avg_attn.cpu()

        return pc_self_attn, pc_cross_attn, avg_attn


    def sample_from_model(self, img, attn = True):
        samples = None
        k = 0
        x_set = []
        frames = []
        pc_self_attns, pc_cross_attns, avg_attns = [], [], []
        with nopdb.capture_call(self.base_model.backbone.resblocks[-1].attn.attention.forward) as attn_call:
            for x in tqdm(self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
                samples = x
                
                if k in self.breakpoints:
                    x_set.append(x)
                    
                    if x.shape[2] == 1024:
                        pc = self.sampler.output_to_point_clouds(samples)[0]
                        pc_self_attn, pc_cross_attn, avg_attn = self.est_attention(attn_call)
                        pc_self_attns.append(pc_self_attn)
                        pc_cross_attns.append(pc_cross_attn)
                        avg_attns.append(avg_attn)
                k += 1    
        
        return pc_self_attns, pc_cross_attns, avg_attns, x_set
    
    def plot_heatmap(self, attn, path):
        for k in range(len(attn)):    
            ax = sns.heatmap(attn[k], cmap = 'rocket_r')
            plt.title('itterations = ' + str(self.breakpoints_downsample[k]))
            plt.savefig(path +str(self.breakpoints_downsample[k]) + '.png', bbox_inches='tight')
            plt.close()
    
    def make_gif(self, path, gif_name, fp = 0.5):
        images = [path + str(k) + '.png' for k in self.breakpoints_downsample]
        frames = []
        for t in self.time_downnsample:
            image = imageio.v2.imread(images[t])
            frames.append(image)   
        imageio.mimsave(path + '/' + gif_name + '.gif', frames, fps = fp)


    def heatmap2d(self, self_attn, cross_attn, avg_attn):
        """
        Make 2D heatmap
        """

        frames = []
        path = self.make_directory('Heatmaps_2D/')
        path_self = self.make_directory('Heatmaps_2D/Self_attention/')
        path_cross = self.make_directory('Heatmaps_2D/Cross_attention/')
        path_full = self.make_directory('Heatmaps_2D/Full_attention/')

        # Make heatmaps
        self.plot_heatmap(self_attn, path_self)
        self.plot_heatmap(cross_attn, path_cross)
        self.plot_heatmap(avg_attn, path_full)

        # Make Gifs
        self.make_gif(path_full, 'full_2d_attention')
        self.make_gif(path_self, 'self_2d_attention')
        self.make_gif(path_cross, 'cross_2d_attention')

    def heatmap2d_inspection(self, attn, i = 250, j = 270):
        pc_self_attn = attn[i:j, i:j]

        pc_self_attn = pc_self_attn.cpu()
        ax = sns.heatmap(pc_self_attn, cmap = 'rocket_r', yticklabels=np.round(np.linspace(i,j,j-i), 0).astype(int), xticklabels=np.round(np.linspace(i,j,j-i).astype(int), 0))
        path = self.make_directory('Heatmaps_2D/inspection/')
        plt.savefig(path + 'zoom.png', bbox_inches='tight')


    def heatmap3d(self, samples, self_attn):
        path = self.make_directory('Heatmaps_3D/')
        frames = []
        for k in range(len(self.breakpoints_downsample)):
            pc = self.sampler.output_to_point_clouds(samples[k])[0]
            
            fig = plot_attention_index(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)), col = self_attn[k])
            fig.suptitle('iterations = ' + str(self.breakpoints_downsample[k]))
            plt.savefig(path + '/fig' + str(self.breakpoints_downsample[k]) + '.png', bbox_inches='tight')
            image = imageio.v2.imread(path + '/fig' + str(self.breakpoints_downsample[k]) + '.png')
            frames.append(image)
            
            
            
        imageio.mimsave(path + '/attention_idx.gif', # output gif
                        frames,          # array of input frames
                        fps = 0.5)         # optional: frames per second
        
    
    def spectra_plots(self, attn):
        frames = []
        path = self.make_directory('Spectra/')
        for k in range(len(self.breakpoints_downsample)):
            pc_self_attn = attn[k]
            fig,a =  plt.subplots(2,2)

            fig.suptitle('iterations = ' + str(self.breakpoints_downsample[k]))
            a[0][0].plot(pc_self_attn[0])
            a[0][0].set_title('i = 0')
            a[0][1].plot(pc_self_attn[500])
            a[0][1].set_title('i = 500')
            a[1][0].plot(pc_self_attn[1000])
            a[1][0].set_title('i = 1000')
            a[1][1].plot(pc_self_attn[-1])
            a[1][1].set_title('i = 1024')

            plt.tight_layout()
            plt.savefig(path + '/fig' + str(self.breakpoints_downsample[k]) + '.png', bbox_inches='tight')
            image = imageio.v2.imread(path + '/fig' + str(self.breakpoints_downsample[k]) + '.png')
            frames.append(image)

        imageio.mimsave(path + '/attention_idx.gif', # output gif
                frames,          # array of input frames
                fps = 0.5)         # optional: frames per second


    def attention_pointcloud(self, samples, self_attn, attn_vals = False, point_tracker = False, max_attention = False, upsample = False):
        frames = []

        if upsample:
            if attn_vals or point_tracker or max_attention:
                print('Can not plot attention cloud with upsample')
                exit 
            else:
                for k in range(len(self.breakpoints)):
                    pc = self.sampler.output_to_point_clouds(samples[k])[0]
                    fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
                    path = self.make_directory('PC_diffusion/')
                    
                    fig.suptitle('iterations = ' + str(self.breakpoints[k]))
                    plt.savefig(path + 'fig' + str(self.breakpoints[k]) + '.png', bbox_inches='tight')
                    image = imageio.v2.imread( path + 'fig' + str(self.breakpoints[k]) + '.png')
                    frames.append(image)

        else: 
            for k in range(len(self.breakpoints_downsample)):
                pc = self.sampler.output_to_point_clouds(samples[k])[0]
                
                if attn_vals:
                    c = np.diagonal(self_attn[k])
                    fig = plot_attention_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)), col = c)
                    path = self.make_directory('PC_with_attention/')

                elif point_tracker:
                    c = np.zeros(1024)
                    fig = plot_attention_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)), col = c, alpha_val = 0.2, track_idx = 500)
                    path = self.make_directory('Point_tracker/')
                
                elif max_attention:
                    c = torch.max(self_attn, dim = 1)[0]
                    fig = plot_attention_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)), col = c)
                    path = self.make_directory('Max_attention/')
                
                fig.suptitle('iterations = ' + str(self.breakpoints_downsample[k]))
                plt.savefig(path + 'fig' + str(self.breakpoints_downsample[k]) + '.png', bbox_inches='tight')
                image = imageio.v2.imread( path + 'fig' + str(self.breakpoints_downsample[k]) + '.png')
                frames.append(image)
                
            
            
        imageio.mimsave(path + 'Attention_cloud.gif', # output gif
                        frames,          # array of input frames
                        fps = 0.5)         # optional: frames per second
        

    def CLIP_embeddings(self, cross_attn, img):
        frames = []
        path = self.make_directory('CLIP_cross_attention/')
        for i,cross in enumerate(cross_attn):
            c = torch.mean(cross, dim= 1)
            sftm = torch.nn.Softmax(dim=0)
            c  = sftm(c)

            c = c[1:257]

            c = c.reshape(1,1,16,16)
        
            c = F.interpolate(c,  size=(img.size[0],img.size[1]), mode= 'bilinear')
            plt.axis('off')
            fig = plt.imshow(c.squeeze(), cmap = 'rocket_r')
            plt.savefig(path + 'fig' + str(i) + '.png', bbox_inches='tight')
            attention_img = Image.open(path + 'fig' + str(i) + '.png').convert(img.mode)
            img = img.resize((500,500))
            attention_img = attention_img.resize((500,500))
            outpict = Image.blend(img,attention_img, 0.5)
            fig = plt.imshow(outpict)
            plt.savefig(path + 'blended_fig'+ str(i) + '.png', bbox_inches='tight', dpi = 300)
            frames.append(outpict) 

        imageio.mimsave(path + 'cross_attn_Diffusion_chair.gif', # output gif
                frames,          # array of input frames
                duration = 0.1)   
    

"""
    def build_off_diagonal_mask(off_element):
        mask = np.zeros_like(pc_self_attn)
        k = 0
        for i in range(len(mask)):
            try:
                mask[i+off_element, i] = 1
            except IndexError:
                pass
        for i in reversed(range(off_element)):
            mask[k, -(i+1)] = 1
            k += 1
        return np.sum(mask*pc_self_attn.numpy(), axis = 1)


    print(build_off_diagonal_mask(0))
    neighbourhood = np.arange(-100, 100, 5)
    print(neighbourhood)
    time = [i for i in range(len(neighbourhood))]

    importlib.reload(point_e.util.plotting)
    from point_e.util.plotting import plot_attention_cloud

    frames = []
    import numpy as np
    for k in neighbourhood:
        c = build_off_diagonal_mask(k)
        #c = np.diag(pc_self_attn)
        #c[500] = 1
        fig = plot_attention_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)), col = c, alpha_val = 0.5)
        fig.suptitle('off_diagonal_element = ' + str(k))
        plt.savefig('Figures/Disco_box/fig' + str(k) + '.png', bbox_inches='tight')
        image = imageio.v2.imread('Figures/Disco_box/fig' + str(k) + '.png')
        frames.append(image)
        
        
        
    imageio.mimsave('Figures/Disco_box/Diffusion_disco.gif', # output gif
                    frames,          # array of input frames
                    fps = 0.5)         # optional: frames per second

"""