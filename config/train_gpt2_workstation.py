# Specialization to my personal workstation paths

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = experiment_name = wandb_run_name = 'gpt2-124M-normalized_gpt_illya_implementation'
wandb_log = True
wandb_notes = "Base normalized GPT run"
wandb_project = "normalized_gpt_dev_sakle"
data_root_path='/mnt/data/'
dataset = 'nanoGPTopenweb'

gradient_accumulation_steps = 1 # 5 * 8
