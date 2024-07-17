import subprocess
formatted_time = subprocess.check_output(['date', '+%Y%m%d%H%M%S']).decode('utf-8').strip()

# Define the paths and other arguments
output_dir = f'output/{formatted_time}/'
model_name_or_path = '/teamspace/studios/this_studio/Language-Model-STS-CFT/Qwen2-0.5B-Instruct' # path of pre-trained model
train_data_path = '/teamspace/studios/this_studio/Language-Model-STS-CFT/data/preprocessed' # train data path
config_file = '/teamspace/studios/this_studio/Language-Model-STS-CFT/train/configs/ddp_config.yaml' #system confige (fixed for single gpu system) 

# Ensure output directory exists
import os
os.makedirs(output_dir, exist_ok=True)

# Run the accelerate launch command
command = f"""
accelerate launch --config_file {config_file} /teamspace/studios/this_studio/Language-Model-STS-CFT/train/train_all.py \
--output_dir {output_dir} \
--model_name_or_path {model_name_or_path} \
--temperature 0.05 \
-- alpha 0.98 \
-- lamb 2.0 \
--train_data_path {train_data_path} \
--learning_rate 5e-5 \
--per_device_train_batch_size 16 \
--bf16 \
--gradient_accumulation_steps 1 \
--warmup_steps 100 \
--max_steps 50000 \
--weight_decay 1e-4 \
--lr_scheduler_type "cosine" \
--save_strategy steps --save_steps 1000 --seed 7 \
--remove_unused_columns False \
--log_level info --logging_strategy steps --logging_steps 10 --report_to wandb 

"""

# Execute the command
import subprocess
subprocess.run(command, shell=True, check=True)
