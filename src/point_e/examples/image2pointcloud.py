# %%
from PIL import Image
import torch
import nopdb
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M' # use base300M or base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

# %%
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
)

# %%
# Load an image to condition on.
img = Image.open('example_data/cube_stack.jpg')

# Produce a sample from the model.
samples = None
with nopdb.capture_calls(base_model.backbone.resblocks[-1].attn.attention.forward) as attn_call:
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x

# %%
print(len(attn_call))
print(attn_call[0].locals.keys())
print(attn_call[0].locals['weight'].shape)
# print(attn_call.locals['x'].shape)

# attentions = attn_call[0].locals['weight'][0, :, 0, 257:].reshape(8, -1)
# print(attentions.shape)
print(attn_call[0].locals['qkv'].shape)


# %%
print(attn_call.locals['q'].shape)
print(attn_call.locals['k'].shape)
print(attn_call.locals['v'].shape)
print(attn_call.locals['weight'].shape)
batch, heads, target, source = attn_call.locals['weight'].shape

# %%
# import math
# scale = 1 / math.sqrt(math.sqrt(attn_call.locals['attn_ch']))

# def reshape_heads_to_batch_dim(tensor):
#         batch_size, seq_len, heads, dim = tensor.shape
#         tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * heads, seq_len, dim)
#         return tensor
# #Manually extract the query and key tensors and combine them as in transformer module, to obtain the attention map.
# new_q = reshape_heads_to_batch_dim(attn_call.locals['q'])
# new_k = reshape_heads_to_batch_dim(attn_call.locals['k'])
# attention_scores = torch.einsum("b i d, b j d -> b i j", new_q, new_k) * scale

# attention_probs = attention_scores.softmax(dim=-1)
# print(attention_probs.shape)

# %%
# pc = sampler.output_to_point_clouds(samples)[0]
# fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))


