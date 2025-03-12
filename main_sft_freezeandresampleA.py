import copy
import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

def compute_rank(matrix):
    return torch.linalg.matrix_rank(matrix).item()

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
print('dataload finished!')

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
print('==================')
print('sample_num_list:')
print(sample_num_list)

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)

fed_args.resample_A = True

for name, param in model.named_parameters():
    if "lora_A" in name:
        param.requires_grad = False  # Freeze A

for name, param in model.named_parameters():
    if "lora_B" in name:
        param.requires_grad = True  # Allow B to update

model.print_trainable_parameters()

model.config.use_cache = False

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

# ===== Define the formatting function =====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
b_norm_history = []
ba_rank_history = []
b_rank_history = []
ba_rank_history_layer_0 = []
ba_rank_history_layer_30 = []
b_rank_history_layer_0 = []
b_rank_history_layer_30 = []


for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)
    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            training_loss[client].append(-1)
            continue

        set_peft_model_state_dict(model, global_dict)
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
        training_args = get_training_args(script_args, new_lr)

        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))

    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list,
        clients_this_round, round, proxy_dict=proxy_dict,
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)
 
 
    with torch.no_grad():
        if (round + 1) % 10 == 0:  # 每 10 轮计算一次
            ba_ranks_layer_0 = []
            b_ranks_layer_0 = []
            ba_ranks_layer_30 = []
            b_ranks_layer_30 = []

            for name, param in model.named_parameters():
                if "lora_B" in name:
                    # 仅计算第 0 层和第 30 层
                    if "layers.0" in name or "layers.30" in name:
                        replaced_key = name.replace("lora_B", "lora_A").replace(".default", "")
                        if replaced_key in global_dict:
                            ba_rank = compute_rank(param @ global_dict[replaced_key])
                            if "layers.0" in name:
                                ba_ranks_layer_0.append(ba_rank)
                            elif "layers.30" in name:
                                ba_ranks_layer_30.append(ba_rank)
                        else:
                            print(f"Key {replaced_key} not found in global_dict")
                        
                        b_rank = compute_rank(param)
                        if "layers.0" in name:
                            b_ranks_layer_0.append(b_rank)
                        elif "layers.30" in name:
                            b_ranks_layer_30.append(b_rank)

            # 计算均值并保存
            if ba_ranks_layer_0:
                ba_rank_history_layer_0.append(np.mean(ba_ranks_layer_0))
                b_rank_history_layer_0.append(np.mean(b_ranks_layer_0))
                np.save(os.path.join(script_args.output_dir, "ba_rank_history_layer_0.npy"), np.array(ba_rank_history_layer_0))
                np.save(os.path.join(script_args.output_dir, "b_rank_history_layer_0.npy"), np.array(b_rank_history_layer_0))
                print(f"Round {round + 1}: Layer 0 - Mean BA Rank: {np.mean(ba_ranks_layer_0)}, Mean B Rank: {np.mean(b_ranks_layer_0)}")

            if ba_ranks_layer_30:
                ba_rank_history_layer_30.append(np.mean(ba_ranks_layer_30))
                b_rank_history_layer_30.append(np.mean(b_ranks_layer_30))
                np.save(os.path.join(script_args.output_dir, "ba_rank_history_layer_30.npy"), np.array(ba_rank_history_layer_30))
                np.save(os.path.join(script_args.output_dir, "b_rank_history_layer_30.npy"), np.array(b_rank_history_layer_30))
                print(f"Round {round + 1}: Layer 30 - Mean BA Rank: {np.mean(ba_ranks_layer_30)}, Mean B Rank: {np.mean(b_ranks_layer_30)}")

    if fed_args.resample_A:
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.copy_(torch.randn_like(param) * 0.01)
    
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

plt.figure()
plt.plot(range(1, fed_args.num_rounds + 1), ba_rank_history, label='Rank of BA')
plt.plot(range(1, fed_args.num_rounds + 1), b_rank_history, label='Rank of B')
plt.xlabel('Communication Round')
plt.ylabel('Rank')
plt.legend()
plt.title('Rank of BA and B over Communication Rounds')
plt.savefig(os.path.join(script_args.output_dir, "rank_plot.png"))
