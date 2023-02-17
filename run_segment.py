import itertools
import os
import random
import time

def run_lora_experiment(param_grid, cmd, n=1000, test = 0, dirname = "lora_grid_search_02_xander", seed = None):
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

  # shuffle the grid values ordering:
  random.shuffle(grid_values)
  already_done = []

  # Loop over the grid values and execute the Python job with each combination of input arguments
  for i, values in enumerate(grid_values[:n]):
    if values in already_done: #This combo has already been tried, skip..
      continue

    already_done.append(values.copy())

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

export CUDA_VISIBLE_DEVICES=2
conda activate diffusers
cd /home/xander/Projects/cog/lora
python run_segment.py

"""

python_cmd = "python lora_diffusion/preprocess_files.py"
input_dir = "/home/xander/Pictures/Mars2023/people/run_segment"

for subdir in sorted(os.listdir(input_dir)):

  full_input_dir = os.path.join(input_dir, subdir) + "/imgs"
  output_dir = os.path.join(input_dir, subdir) + "/train"

  param_grid = {
    'files': full_input_dir,
    'output_dir': output_dir,
    'target_prompts': "face",
    'target_size': 512,
  }

  run_lora_experiment(param_grid, python_cmd)