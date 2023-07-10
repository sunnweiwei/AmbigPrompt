import sys
import os


# Train DPR with https://github.com/luyug/Dense
sys.path += [r'./', r'/my/path/to/dense/src']
from dense.driver.train import main

output_dir = f'out/dense'
luyu = 'Luyu/co-condenser-wiki'
os.makedirs(output_dir, exist_ok=True)
nq_args = f'''
--output_dir {output_dir} 
--model_name_or_path {luyu}
--do_train 
--save_steps 5000 
--logging_steps 100
--train_dir ./dense/examples/wikipedia-nq/nq-train/bert
--per_device_train_batch_size 8
--train_n_passages 3
--learning_rate 1e-3
--q_max_len 32
--p_max_len 156
--num_train_epochs 2
--overwrite_output_dir
--dataloader_num_workers 5
'''
# custom_args += f' --train_n_passages {train_n_passages} '
# custom_args = '--fp16 '
custom_args = ''
custom_args += ' --adafactor true '
custom_args += ' --negatives_x_device '
args = (' '.join([nq_args, custom_args])).split()

sys.argv += args
print(args)

main()
