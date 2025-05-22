import os
import math
import json
import time
from typing import Callable
from collections import OrderedDict
from torch import Tensor, svd_lowrank as fast_svd, block_diag

from common.lora_modules.lora import *
from common.utils.utils import Timer, reduce_tensor, to_device, print_rank_0, ensure_directory_exists

def get_est_nuc_norm(tensor, rank):
    _, Sr, _ = fast_svd(tensor, rank, niter=8)
    return torch.sum(torch.log1p(Sr))

class LinearWithRaLoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        Ralora_dynamic_scaling: bool = False,
        forward_method: str = 'for'
    ):  
        self.dynamic_scaling = Ralora_dynamic_scaling
        self.scaling_alpha = lora_config.lora_scaler
        super().__init__(lora_config)

        if lora_config.quant:
            print(f'Currently RaLoRA is incompatible with quant, skipped quant')
        self.lora_config = lora_config
        
        if forward_method == "for":
            print(f"init_lora_weights method: {forward_method}")
            self.init_lora_weights = self.init_lora_weights_for
            self._lora_forward = self._lora_forward_for
        elif forward_method == "einsum":
            print(f"init_lora_weights method: {forward_method}")
            self.init_lora_weights = self.init_lora_weights_einsum
            self._lora_forward = self._lora_forward_einsum
        # Dynamic n allocation parameters

    def _prepare_Ralora_attrs(self, lora_rank, in_features, out_features):
        
        self.lora_rank = lora_rank * self.n_split    
        self.in_features = in_features
        self.out_features = out_features

        self._check_exact_division()
        self.mini_lora_rank = int(lora_rank)
        self.mini_in_features = int(self.in_features / self.n_split)
        self.mini_out_features = int(self.out_features / self.n_split)

    def _check_exact_division(self):
        if self.in_features % self.n_split != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by melora_n_split ({self.n_split})")
        if self.out_features % self.n_split != 0:
            raise ValueError(f"out_features ({self.out_features}) must be divisible by melora_n_split ({self.n_split})")

    def init_Ralora_weights(self):
        self.lora_rank = 0
        
    def init_lora_weights_for(self):
        print(f"init ralora weight for")
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_a, self.weight_b =nn.ParameterList(), nn.ParameterList()  
        for _ in range(self.n_split):
            mini_weight_a = nn.Parameter(torch.empty((self.mini_lora_rank, self.mini_in_features), dtype=dtype), requires_grad=requires_grad)
            mini_weight_b = nn.Parameter(torch.zeros((self.mini_out_features, self.mini_lora_rank), dtype=dtype), requires_grad=requires_grad)
            self.weight_a.append(mini_weight_a)
            self.weight_b.append(mini_weight_b)
        self._init_weight('weight_a')
        self._init_weight('weight_b')

    # Using einsum to speed up
    def init_lora_weights_einsum(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_a = nn.Parameter(torch.empty((self.n_split, self.mini_lora_rank, self.mini_in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.zeros((self.n_split, self.mini_out_features, self.mini_lora_rank), dtype=dtype), requires_grad=requires_grad)
        super()._init_weight('weight_a')
        super()._init_weight('weight_b')

    def _init_weight(self, weight_name: str):
        weight_list = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        for weight in weight_list:
            self.get_weight_init_method(**init_kwargs)(weight)

    def _lora_forward_for(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        if self.lora_rank == 0:
            return result
        else:
            device = x.device
            dtype = self._get_lora_dtype()
            weight_a = self._diagonal_concat_weight_a().to(dtype=dtype, device=device)
            weight_b = self._diagonal_concat_weight_b().to(dtype=dtype, device=device)
            lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b).to(result.dtype)
            return result + self.lora_scaler * lora_result
    
    def _lora_forward_einsum(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        if self.lora_rank == 0:
            return result
        else:
            bsz_seq_len = x.shape[:-1]
            x = x.view(*bsz_seq_len, self.n_split, self.mini_in_features)
            xa = torch.einsum("...si,sri->...sr", self.lora_dropout(x), self.weight_a.to(self._get_lora_dtype()))
            lora_result = torch.einsum("...sr,sor->...so", xa, self.weight_b.to(self._get_lora_dtype())).reshape(*bsz_seq_len, self.out_features)
            return result + self.lora_scaler * lora_result

    def _compute_lora_weight(self):
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._diagonal_concat_weight_a().to(self._get_lora_dtype())
            weight_b = self._diagonal_concat_weight_b().to(self._get_lora_dtype())
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight.to(self.weight.dtype)

    def _diagonal_concat_weight_a(self):
        return block_diag(*self.weight_a)

    def _diagonal_concat_weight_b(self):
        return block_diag(*self.weight_b)
                                                            
    def _get_scaling(self, avg_rank, real_rank):
        if self.dynamic_scaling:
            self.scale_rank = real_rank
        else:
            self.scale_rank = avg_rank

        self.lora_scaler = self.scaling_alpha / self.scale_rank
    
    def dynamic_init(self, avg_rank, rank, n_split=None):
        """
        During inference, this should be called before loading checkpoint, 
        and set the init method to vanilla
        """
        # If dynamic n allocation is enabled and n_split is provided, use it to override the default
        if n_split >= 1:
            self.n_split = n_split
        else:
            raise ValueError(f"The value of n_split: {n_split} must be greater than or equal to 1")
            
        self._prepare_Ralora_attrs(rank, 
                                   self.lora_config.in_features, 
                                   self.lora_config.out_features)
        
        if rank != 0:
            self._get_scaling(avg_rank, rank)
            with torch.no_grad():
                self.init_lora_weights()
            if hasattr(self.weight, "grad_stored"):
                del self.weight.grad_stored
            if hasattr(self.weight, "iters"):
                del self.weight.iters
            
        

    # def compress_init(self, 
    #                 #   niters, 
    #                   lr: float, 
    #                   scaling_by_lr: bool = False, 
    #                   stable_gemma: int = 8, 
    #                   reinit_weight: bool = False,
    #                   weight_init_a: bool = False,
    #                   grad_init_a: bool = False):
    #     pass
               
    # def grad_svd_init(self, 
    #                  direction: str = 'ArB2r', 
    #                  scale: str = 'stable', 
    #                  stable_gamma: int = 16, 
    #                  scaling_factor: int = 16):
    #     pass
    
    # def weight_svd_init(self):
    #     pass
       
def get_record_gradient_hook(model):
    def record_gradient_hook(grad):
        torch.cuda.synchronize()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                if not hasattr(p, 'grad_stored'):
                    p.grad_stored = p.grad.detach().cpu()
                    p.iters = 1
                else:
                    p.grad_stored += p.grad.detach().cpu()
                    p.iters += 1
                p.grad = None
        return grad

    return record_gradient_hook

def compute_importance(param, grad_stored):
    param = param.float()
    grad_stored = grad_stored.float().to(param.device)
    importance = torch.mean(torch.abs(param * grad_stored)).item()
    return isinstance(importance, tuple), importance
    
def get_normalized_importances(importances_tensor):

    normalized_importances = importances_tensor / importances_tensor.sum()

    return normalized_importances


def get_allocated_rank(model, args):
    named_ranks = {}
    named_importances = OrderedDict()
    total_budget, smooth_total_budget, actual_trainable = 0, 0, 0
    named_features, named_smooth_features = {}, {}

    feature_adjust_func: Callable = {
        'sqrt': math.sqrt,
        'log1p': math.log1p,
        None: lambda x: x  
    }.get(args.Ralora_features_func, lambda x: x)

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LinearWithRaLoRA):
                # Calculate the importance
                if not hasattr(module.weight, 'grad_stored'):
                    print_rank_0(f'--->Module: {name} do not have stored gradients', args.global_rank)
                    continue
                features = module.in_features + module.out_features
                is_tuple, importance = compute_importance(module.weight.data, 
                                                module.weight.grad_stored)
                named_importances[name] = importance

                # Calculate features and budget
                # 可以减少features对rank分配的影响，但是似乎是负面影响？
                adjusted_features = feature_adjust_func(features)
                named_smooth_features[name] = adjusted_features
                named_features[name] = features

                smooth_total_budget += adjusted_features * args.lora_rank
                total_budget += features * args.lora_rank

        if not named_importances:
            raise ValueError("No gradients were stored. Check if backward pass was performed correctly.")

        # Calculate softmax of importances
        if is_tuple:
            first_importances_tensor = torch.tensor([i[0] for i in list(named_importances.values())])
            second_importances_tensor = torch.tensor([i[1] for i in list(named_importances.values())])
            first_normalized_importances = get_normalized_importances(first_importances_tensor)
            second_normalized_importances = get_normalized_importances(second_importances_tensor)
            normalized_importances = torch.tensor([0.5 * a + 0.5 * b for a, b in zip(first_normalized_importances, 
                                                                                     second_normalized_importances)])
        else:
            importances_tensor = torch.tensor(list(named_importances.values()))
            normalized_importances = get_normalized_importances(importances_tensor)

        # Allocate ranks based on calculated budgets
        for name, normalized_importance in zip(named_importances.keys(), normalized_importances):
            # 均衡的问题
            smooth_trainable = round(smooth_total_budget * normalized_importance.item())

            rank = smooth_trainable // named_smooth_features[name]
            if args.Ralora_max_rank and args.Ralora_min_rank:
                named_ranks[name] = min(max(round(rank), args.Ralora_min_rank), args.Ralora_max_rank)
            actual_trainable += named_ranks[name] * named_features[name]

    return total_budget, actual_trainable, named_ranks, named_importances

def compute_effective_rank(gradient_matrix, dtype=torch.float32, eps=1e-10):
    """
    Compute the effective rank of a gradient matrix using the method from
    "THE EFFECTIVE RANK: A MEASURE OF EFFECTIVE DIMENSIONALITY" (Roy & Vetterli, 2007).

    Args:
        gradient_matrix (torch.Tensor): Input gradient matrix (2D tensor, shape: [m, n]).
        eps (float): Small value to avoid numerical instability in log computation (default: 1e-10).

    Returns:
        float: Effective rank of the gradient matrix.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradient_matrix = gradient_matrix.to(dtype=dtype, device=device)
    # Ensure the input is a 2D tensor
    if gradient_matrix.dim() != 2:
        raise ValueError("Input gradient_matrix must be a 2D tensor")

    # Perform Singular Value Decomposition (SVD)
    try:
        U, S, Vh = torch.linalg.svd(gradient_matrix)
    except RuntimeError as e:
        print(f"SVD computation failed: {e}")
        return 1.0  # Return minimal effective rank in case of failure

    # If no valid singular values, return minimal effective rank
    if S.numel() == 0:
        print('Some thing wrong, because the number of S=0')
        return 1.0

    # Compute L1 norm of singular values
    l1_norm = torch.sum(S)

    # Compute normalized singular values (p_k = sigma_k / ||sigma||_1)
    p = S / l1_norm

    # Compute Shannon entropy: H = -sum(p_k * log(p_k))
    # Add eps to avoid log(0)
    entropy = -torch.sum(p * torch.log(p + eps))

    # Compute effective rank: erank = exp(H)
    effective_rank = torch.exp(entropy).item()
    
    del U, S, Vh, gradient_matrix
    # Ensure effective rank is at least 1
    return max(1.0, effective_rank)

def compute_n_split_allocations(model, named_ranks, args):
    """
    Compute the optimal number of mini LoRA modules for each layer
    based on gradient importance.
    
    Returns a dictionary mapping module names to the number of 
    mini LoRA modules to use for that layer.
    """

    named_n_splits = {}
    named_grad_importants = {}
    assert args.erank_max_power >= 0, f'The eran_max_power must be setted.'
    print_rank_0(f'--->Allocate n using the erank method.', args.global_rank)
    print_rank_0(f'--->The max n is {args.erank_max_power}.', args.global_rank)
    min_power = 0
    max_power = args.erank_max_power
    start_time = time.time()
    for name, module in model.named_modules():
        if isinstance(module, LinearWithRaLoRA):
            if not hasattr(module.weight, 'grad_stored'):
                    print_rank_0(f'--->Module: {name} does not have stored gradients', args.global_rank)
                    continue
            erank = compute_effective_rank(module.weight.grad_stored)
            named_grad_importants[name] = erank
    end_time = time.time()
    print_rank_0(f'--->Time consumption for calculating svd: {end_time-start_time:.6f}s', args.global_rank)
    if not named_grad_importants:
            print_rank_0(f'--->No gradient erank calculated for dynamic n allocation', args.global_rank)
            return {}

    # Allocating n according erank and lora rank
    for name, erank in named_grad_importants.items():
        n_splits_power = min(max_power, max(min_power, math.floor(math.log2(erank) - math.log2(named_ranks[name]))))
        named_n_splits[name] = 2 ** n_splits_power
        print_rank_0(f'--->Module {name}: grad_erank={erank:.6f},  n_split={named_n_splits[name]}', args.global_rank)

    return named_n_splits

def RaLoRA_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
    task_name: str = '',
    forward_backward_func: Callable = None
):
    print_rank_0("--->Estimating gradient for RaLoRA.", rank=args.global_rank)
    with Timer() as timer:
        model.to(args.device)
        model.train()

        # Note that we only compute gradient for RaLoRA layers.
        # Avoiding unnecessary computing.
        hooks = [
            module.weight.register_hook(get_record_gradient_hook(model))
            for module in model.modules()
            if isinstance(module, LinearWithRaLoRA)
        ]

        for module in model.modules():
            if isinstance(module, LinearWithRaLoRA):
                module.weight.requires_grad = True

        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            if forward_backward_func:
                loss = forward_backward_func(model, batch)
            elif args.huggingface:
                loss = model(input_ids=batch['input_ids'],
                            labels=batch['labels'],
                            attention_mask=batch['attention_mask']).loss
            else:
                output = model(**batch)
                loss = output[0]
            loss.backward()
            print_rank_0(f'--->RaLoRA gradient computing step: {idx+1}, loss: {loss.item()}, remaining steps: {iters - (idx+1)} ', args.global_rank)

            if (idx + 1) == iters:
                break

        for hook in hooks:
            hook.remove()

        for p in model.parameters():
            p.grad = None
            
        if args.world_size > 1:
            torch.distributed.barrier()

        print_rank_0('--->All reduce RaLoRA stored gradients if needed.', args.global_rank)
        for p in model.parameters():
            if hasattr(p, 'grad_stored'):
                p.grad_stored /= p.iters
                if args.world_size > 1:
                    p.grad_stored = reduce_tensor(p.grad_stored.to(args.device), args.world_size).to('cpu')

        total_budget, actual_trainable, named_ranks, named_importances = get_allocated_rank(model, args)
        
        # Compute and allocate optimal number of mini LoRA modules
        named_n_splits = {}
        print_rank_0('--->Computing dynamic n allocation for ME-LoRA', args.global_rank)
        named_n_splits = compute_n_split_allocations(model, named_ranks, args)

        save_floder = os.path.join(args.output_path, args.experiment_name)
        if task_name:
            save_floder = os.path.join(save_floder, task_name)
            
        ensure_directory_exists(save_floder, args.global_rank)
        if args.global_rank == 0:
            with open(os.path.join(save_floder, 'rank.json'), 'w') as f:
                json.dump(named_ranks, f)
            with open(os.path.join(save_floder, 'importance.json'), 'w') as f:
                json.dump(named_importances, f)
            if named_n_splits:
                with open(os.path.join(save_floder, 'n_splits.json'), 'w') as f:
                    json.dump({name: int(n) for name, n in named_n_splits.items()}, f)

        print_rank_0(f'--->RaLoRA total budget: {total_budget}, actual trainable: {actual_trainable}', args.global_rank)
        for name, module in model.named_modules():
            if isinstance(module, LinearWithRaLoRA) and name in named_ranks.keys():
                n_split = named_n_splits.get(name, 1)
                print_rank_0(f'--->Module {name} is initiating lora weight, rank: {named_ranks[name]}, n_split: {n_split}', args.global_rank)
                module.dynamic_init(args.lora_rank, named_ranks[name], n_split=n_split)
        torch.cuda.empty_cache()

    print_rank_0(f'--->Total time consumed for RaLoRA initialization: {timer.time_cost}', args.global_rank)