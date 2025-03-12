import torch
import copy
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict

def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif fed_args.fed_alg == 'gd': 
        trainer = SFTTrainerGD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            learning_rate=script_args.learning_rate,  # 使用 script_args 中的学习率
        )
    elif fed_args.fed_alg == 'sgdr':  # 支持 SGDr
        trainer = SFTTrainerSGDr(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            learning_rate=script_args.learning_rate,  
            rank=script_args.peft_lora_r,  
            reg=1e-6, 
        )
    elif fed_args.fed_alg == 'adamwr': 
        trainer = SFTTrainerAdamWr(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            rank=script_args.peft_lora_r, 
            reg=1e-6,  
        )
    elif (fed_args.fed_alg in ['fedavg', 'fedavgm', 'fedadgrad', 'fedyogi', 'fedadam']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

######################
class SFTTrainerAdamWr(SFTTrainer):
    def __init__(self, rank=2, reg=1e-6, **kwargs):
        super(SFTTrainerAdamWr, self).__init__(**kwargs)
        self.rank = rank  
        self.reg = reg  
        self.states = {}  
        if self.optimizer is not None:
              print('optimizer is not none')
              self.optimizer.step = lambda: None
        
    def _init_state(self, param):
        state = self.states.get(param, {})
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(param.data)
        if "exp_avg_sq" not in state:
            state["exp_avg_sq"] = torch.zeros_like(param.data)
        if "step" not in state:
            state["step"] = 0
        self.states[param] = state
        return state

    def _update_param(self, param, grad, state, group):
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        state["step"] += 1

       
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

       
        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]

       
        denom = exp_avg_sq.sqrt().add_(group["eps"])
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

       
        param.data.addcdiv_(-step_size, exp_avg, denom)

  
        if group["weight_decay"] > 0.0:
            param.data.add_(param.data, alpha=-group["lr"] * group["weight_decay"])

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

    
        loss = self.compute_loss(model, inputs)
        loss.backward()


        with torch.no_grad():
            named_params = list(model.named_parameters())
            i = 0
            while i < len(named_params):
                name1, p1 = named_params[i]
                if name1.endswith("lora_A.default.weight"): 
                    if i + 1 < len(named_params):
                        name2, p2 = named_params[i + 1]
                        if name2.endswith("lora_B.default.weight") and name2.replace("lora_B.default.weight", "lora_A.default.weight") == name1:
                            
                            if p1.grad is None or p2.grad is None:
                                i += 1
                                continue

                            
                            state_p1 = self._init_state(p1)
                            state_p2 = self._init_state(p2)

                         
                            grad1_0 = p1.grad.data[0:self.rank, :]
                            grad1_1 = p1.grad.data[self.rank:, :]
                            scale1_0 = p2.data[0:dim_1, :]
                            scale1_1 = p2.data[dim_1:, :]

                            try:
                                grad1_0_scaled = torch.inverse(scale1_0.T @ scale1_0 + self.reg * torch.eye(self.rank)).to(scale1_0.device) @ grad1_0
                            except:
                                grad1_0_scaled = grad1_0

                            try:
                                grad1_1_scaled = torch.inverse(scale1_1.T @ scale1_1 + self.reg * torch.eye(self.rank)).to(scale1_1.device) @ grad1_1
                            except:
                                grad1_1_scaled = grad1_1

                            grad1_scaled = torch.cat([grad1_0_scaled, grad1_1_scaled])
                            self._update_param(p1, grad1_scaled, state_p1, group)

                       
                            grad2_0 = p2.grad.data[0:dim_1, :]
                            grad2_1 = p2.grad.data[dim_1:, :]
                            scale2_0 = p1.data[0:self.rank, :]
                            scale2_1 = p1.data[self.rank:, :]

                            try:
                                grad2_0_scaled = grad2_0 @ torch.inverse(scale2_0 @ scale2_0.T + self.reg * torch.eye(self.rank)).to(scale2_0.device)
                            except:
                                grad2_0_scaled = grad2_0

                            try:
                                grad2_1_scaled = grad2_1 @ torch.inverse(scale2_1 @ scale2_1.T + self.reg * torch.eye(self.rank)).to(scale2_1.device)
                            except:
                                grad2_1_scaled = grad2_1

                            grad2_scaled = torch.cat([grad2_0_scaled, grad2_1_scaled])
                            self._update_param(p2, grad2_scaled, state_p2, group)

                            i += 1  
                i += 1

        return loss.detach()


class SFTTrainerSGDr(SFTTrainer):
    def __init__(self, learning_rate, rank=4, reg=1e-6, **kwargs):
        super(SFTTrainerSGDr, self).__init__(**kwargs)
        self.learning_rate = learning_rate  
        self.rank = rank  
        self.reg = reg  
        if self.optimizer is not None:
            print('optimizer is not none')
            self.optimizer.step = lambda: None 

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()

        with torch.no_grad():
            named_params = list(model.named_parameters())
            i = 0
            while i < len(named_params):
                name1, p1 = named_params[i]
                if name1.endswith("lora_A.default.weight"):  # check lora_A
                    if i + 1 < len(named_params):
                        name2, p2 = named_params[i + 1]
                        if name2.endswith("lora_B.default.weight") and name2.replace("lora_B.default.weight", "lora_A.default.weight") == name1:
                            if i < 5:
                                print(f"p1: {name1}, p2: {name2}")
                                print(f"p1 grad is {'not none' if p1.grad is not None else 'none'}")
                                print(f"p2 grad is {'not none' if p2.grad is not None else 'none'}")
                            i += 1
    
                            if p1.grad is None and  p2.grad is None:
                                continue
                            dim_1 = p2.data.shape[0] // 2
                            if p1.grad is not None: 
                                # update p1
                                grad1_0 = p1.grad.data[0:self.rank, :]
                                grad1_1 = p1.grad.data[self.rank:, :]
                                scale1_0 = p2.data[0:dim_1, :]
                                scale1_1 = p2.data[dim_1:, :]

                                try:
                                    grad1_0_scaled = torch.inverse(scale1_0.T @ scale1_0 + self.reg * torch.eye(self.rank)).to(scale1_0.device) @ grad1_0
                                except:
                                    grad1_0_scaled = grad1_0

                                try:
                                    grad1_1_scaled = torch.inverse(scale1_1.T @ scale1_1 + self.reg * torch.eye(self.rank)).to(scale1_1.device) @ grad1_1
                                except:
                                    grad1_1_scaled = grad1_1

                                grad1_scaled = torch.cat([grad1_0_scaled, grad1_1_scaled])
                                p1.data.add_(grad1_scaled, alpha=-self.learning_rate)
                            
                            if p2.grad is not None:
                                # update p2
                                grad2_0 = p2.grad.data[0:dim_1, :]
                                grad2_1 = p2.grad.data[dim_1:, :]
                                scale2_0 = p1.data[0:self.rank, :]
                                scale2_1 = p1.data[self.rank:, :]

                                try:
                                    grad2_0_scaled = grad2_0 @ torch.inverse(scale2_0 @ scale2_0.T + self.reg * torch.eye(self.rank)).to(scale2_0.device)
                                except:
                                    grad2_0_scaled = grad2_0

                                try:
                                    grad2_1_scaled = grad2_1 @ torch.inverse(scale2_1 @ scale2_1.T + self.reg * torch.eye(self.rank)).to(scale2_1.device)
                                except:
                                    grad2_1_scaled = grad2_1

                                grad2_scaled = torch.cat([grad2_0_scaled, grad2_1_scaled])
                                p2.data.add_(grad2_scaled, alpha=-self.learning_rate)
                i += 1
        return loss.detach()




class SFTTrainerGD(SFTTrainer):
    def __init__(self, learning_rate, rank=4, reg=1e-6, **kwargs):
        super(SFTTrainerGD, self).__init__(**kwargs)
        self.learning_rate = learning_rate  
        self.rank = rank  
        self.reg = reg  
        if self.optimizer is not None:
            print('optimizer is not none')
            self.optimizer.step = lambda: None 

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_A.default.weight" in name or "lora_B.default.weight" in name:
                    if param.grad is not None:
                        # Directly update the parameter using gradient descent
                        param.data.add_(param.grad, alpha=-self.learning_rate)
        
        return loss.detach()


##########################

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)