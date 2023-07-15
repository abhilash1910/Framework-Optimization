from transformers.models.gptj.modeling_gptj import GPTJBlock

from parallelformers.policies.base import Layer, Policy
from parallelformers.utils import AllReduceLinear


class GPTJPolicy(Policy):
    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "attn.embed_dim": config.hidden_size // world_size,
            # 2. reduce number of heads
            "attn.num_attention_heads": config.n_head // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(weight="attn.q_proj.weight"),
            Layer(weight="attn.k_proj.weight"),
            Layer(weight="attn.v_proj.weight"),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="attn.out_proj.weight",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="mlp.fc_in.weight",
                bias="mlp.fc_in.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="mlp.fc_out.weight",
                bias="mlp.fc_out.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return GPTJBlock