import itertools
import os
import random
import time

def run_lora_experiment(param_grid, n=1000, test = 0, dirname = "lora_grid_search_02_xander", seed = None):
  if seed is not None:
      random.seed(seed)
  else:
      random.seed(int(time.time()))

  # Split the parameter grid into fixed and variable arguments
  fixed_args = {k: v for k, v in param_grid.items() if isinstance(v, (int, str, bool, float))}
  variable_args = {k: v for k, v in param_grid.items() if k not in fixed_args}

  # Generate all combinations of variable arguments
  variable_values = list(itertools.product(*[v if not isinstance(v, list) else [v] for v in variable_args.values()]))
  variable_keys = list(variable_args.keys())

  # Generate a long list of grid_values by randomly sampling each argument list
  long_grid_values = []
  for i in range(10000):
      values = {}
      for k in variable_keys:
          if isinstance(param_grid[k], list):
              values[k] = random.choice(param_grid[k])
          else:
              values[k] = param_grid[k]
      long_grid_values.append(values)
  
  # Randomly sample a subset of the long list of grid_values
  grid_values = random.sample(long_grid_values, n)

  # Combine fixed and variable arguments into a single dictionary
  grid_values = [{**fixed_args, **values} for values in grid_values]

  # Define the command to execute your Python job with input arguments
  cmd = 'python lora_diffusion/cli_lora_pti.py'

  # shuffle the grid values ordering:
  random.shuffle(grid_values)
  already_done = []

  # Loop over the grid values and execute the Python job with each combination of input arguments
  for i, values in enumerate(grid_values[:n]):
    if values in already_done: #This combo has already been tried, skip..
      continue

    already_done.append(values.copy())

    # get the datadirectory name:
    data_dir = "_".join(values['instance_data_dir'].split('/')[-2:])

    # generate a short, pseudorandom character id for this run:
    id_str = ''.join(random.choice('0123456789abcdef') for i in range(6))

    values['output_dir'] = f"./exps/{dirname}/{data_dir}_{i:02d}_{id_str}"

    arg_str = ' '.join([f'--{k} {v}' for k, v in values.items()])
    full_cmd = f'{cmd} {arg_str}'
    print('------------------------------------------')
    print(f'Running command: {i+1}/{n}')

    # pretty print the values dictionary:
    for k, v in values.items():
      print(f'{k}:{" "*(50-len(k))}{v}')

    if not test:
      os.system(full_cmd)


"""

export CUDA_VISIBLE_DEVICES=3
conda activate diffusers
cd /home/xander/Projects/cog/lora
python grid_train_lora.py


'instance_data_dir':             "/home/xander/Pictures/Mars2023/people/gene/train",
'instance_data_dir':             "/home/xander/Pictures/Mars2023/people/gene/train_one",
'instance_data_dir':             "/home/xander/Pictures/Mars2023/people/niko/train",
'instance_data_dir':             "/home/xander/Pictures/Mars2023/people/gene/train",


"""

n_to_run = 50

param_grid = {
  'pretrained_model_name_or_path': ['dreamlike-art/dreamlike-photoreal-2.0'],
  'instance_data_dir':             "/home/xander/Pictures/Mars2023/people/ready/xander/train",

  'train_text_encoder':            True,
  'perform_inversion':             True,
  'learning_rate_ti':              [1e-4, 2.5e-4],
  'continue_inversion':            True,
  'continue_inversion_lr':         [0.5e-5, 2e-5, 1e-4],
  'learning_rate_unet':            [1.0e-5, 2.5e-5],
  'learning_rate_text':            [1.0e-5, 2.5e-5],
  'save_steps':                    50,
  'max_train_steps_ti':            [200, 300, 400], 
  'max_train_steps_tuning':        [300, 450, 600], 
  'weight_decay_ti':               [0.001, 0.005],
  'weight_decay_lora':             [0.0001, 0.001],
  'lora_rank_unet':                [1,2,4],
  'lora_rank_text_encoder':        [1,4,8,16],
  'use_extended_lora':             [False, True],

  'use_face_segmentation_condition': True,
  'use_mask_captioned_data':       False,
  'placeholder_tokens':            "\"<person1>|<person2>\"",
  'proxy_token':                   "person",
  'use_template':                  "person",
  'clip_ti_decay':                 True,

  'cached_latents':                False,
  'train_batch_size':              4,
  'gradient_accumulation_steps':   1,
  'color_jitter':                  True,
  'scale_lr':                      True,
  'lr_scheduler':                  "linear",
  'lr_warmup_steps':               0,

  'resolution':                    512,
  'enable_xformers_memory_efficient_attention': True,

}

run_lora_experiment(param_grid, n=n_to_run)