import torch
from torch import nn
import torch.nn.functional as F

import math


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        result += (
            self.lora_dropout(x)
            @ self.lora_A.transpose(0, 1)
            @ self.lora_B.transpose(0, 1)
        ) * self.scaling
        return result


def find_and_insert(model, lora_rank=16):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and ("q_proj" in name or "v_proj" in name):
            w = child.weight.data

            bias = False if child.bias is None else True
            layer = Linear(
                child.in_features,
                child.out_features,
                r=lora_rank,
                lora_alpha=lora_rank,
                bias=bias,
            )
            layer.weight.data = w
            if bias:
                layer.bias.data = child.bias.data

            setattr(model, name, layer)
        else:
            find_and_insert(child, lora_rank)


def find_and_merge(model):
    for name, child in model.named_children():
        if isinstance(child, LoRALayer):
            w = child.weight.data
            a = child.lora_A.data
            b = child.lora_B.data

            bias = False if child.bias is None else True
            layer = nn.Linear(w.shape[1], w.shape[0], bias=bias)

            layer.weight.data = w + b @ a
            if bias:
                layer.bias.data = child.bias.data

            setattr(model, name, layer)
        else:
            find_and_merge(child)


def mark_only_lora_as_trainable(model):
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True


def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def get_lora_model(model, lora_rank):
    find_and_insert(model, lora_rank=lora_rank)
    mark_only_lora_as_trainable(model)
    print_trainable_parameters(model)

    return model


def merge_lora_model(model):
    find_and_merge(model)

    return model


def lora_state_dict(model):
    my_state_dict = model.state_dict()
    to_return = {}
    for k in my_state_dict:
        if "lora_" in k:
            to_return[k] = my_state_dict[k]

    return to_return


class Projection:
    def __init__(self, model):
        self.model = model
        self.eigen_vectors = self.decompose_pretrained_weights()

    def decompose_pretrained_weights(self):
        eigen_vectors = {}
        for name, param in self.model.state_dict().items():
            if "q_proj.weight" in name or "v_proj.weight" in name:
                w_zero = param.data
                U, S, VH = torch.linalg.svd(w_zero)
                eigen_vectors[name.replace("weight", "U")] = U[:, :256]
                eigen_vectors[name.replace("weight", "VH")] = VH[:256, :]
                
        return eigen_vectors

    def update(self):
        state_dict = self.model.state_dict()
        for name, param in state_dict.items():
            if "lora_B" in name:
                lora_B = param
                lora_A = state_dict[name.replace("lora_B", "lora_A")]
                delta_w = lora_B @ lora_A

                U = self.eigen_vectors[name.replace("lora_B", "U")]
                VH = self.eigen_vectors[name.replace("lora_B", "VH")]

                S_prime = U.T @ delta_w @ VH.T

                # pick top k eigen values and corresponding eigen vectors
                # _, indices = torch.diag(S_prime).topk(16)
                # state_dict[name] = U[:, indices] @ torch.diag(torch.sqrt(torch.diag(S_prime)[indices]))
                # state_dict[name.replace("lora_B", "lora_A")] = torch.diag(torch.sqrt(torch.diag(S_prime)[indices])) @ VH[indices, :]
                state_dict[name] = U
                state_dict[name.replace("lora_B", "lora_A")] = torch.diag(torch.diag(S_prime)) @ VH

        self.model.load_state_dict(state_dict)


class Projection2:
    def __init__(self, model):
        self.model = model
        self.eigen_vectors = self.decompose_pretrained_weights()
        self.shadow_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def decompose_pretrained_weights(self):
        eigen_vectors = {}
        for name, param in self.model.state_dict().items():
            if "q_proj.weight" in name or "v_proj.weight" in name:
                w_zero = param.data
                U, S, VH = torch.linalg.svd(w_zero)
                eigen_vectors[name.replace("weight", "U")] = U
                eigen_vectors[name.replace("weight", "VH")] = VH
                
        return eigen_vectors
    
    def update(self):
        state_dict = self.model.state_dict()

        for name in self.shadow_params.keys():
            if "lora_B" in name:
                lora_B = self.shadow_params[name]
                lora_A = self.shadow_params[name.replace("lora_B", "lora_A")]
                delta_w = lora_B @ lora_A

                diff_lora_B = state_dict[name] - lora_B
                diff_lora_A = state_dict[name.replace("lora_B", "lora_A")] - lora_A
                diff_delta_w = diff_lora_B @ diff_lora_A

                U = self.eigen_vectors[name.replace("lora_B", "U")]
                VH = self.eigen_vectors[name.replace("lora_B", "VH")]

                S_prime = U.T @ diff_delta_w @ VH.T

                S_prime = torch.diag(torch.diag(S_prime))

                diff_delta_w = U @ S_prime @ VH
                
                P, S, Q = torch.svd_lowrank(diff_delta_w + delta_w, q=16)
                state_dict[name] = P @ torch.sqrt(torch.diag(S))
                state_dict[name.replace("lora_B", "lora_A")] = torch.sqrt(torch.diag(S)) @ Q.T
        self.model.load_state_dict(state_dict)

        self.shadow_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    model = get_lora_model(model)

    projector = Projection2(model)
    # print(projector.eigen_vectors.keys())

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    input_ids = torch.randint(0, 50257, (1, 1024))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 50257, (1, 1024))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    optimizer.step()

    projector.update()