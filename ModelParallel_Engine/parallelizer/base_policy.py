from contextlib import suppress
import transformers
from torch import nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type


@dataclass
class Layer:
    r"""
    Dataclass used to describe a layer in the policy object

    Attributes:
        weight and bias (str): the names of the weight and bias tensors, respectively. You can use the syntax
            such as `[ ]` or `.` to the tensor names. `.` is used as accessors in common programming languages and `[ ]`
            is used to access elements in nn.ModuleList.
        n_fused (int): the number of areas used in fused layers. For example, GPT2 and TransfoXL have fused
            attention layers that consist of query, key and value. These layers should not be simply chunked by
            the number of GPUs. Instead, they should be divided into the query, key and value areas first.
        replace (Any): the layer that you want to replace an existing layer with. The parallelization process
            by the tensor slicing method involves All-Reduce operations to collect tensors from all GPUs.
            So, we need to insert some layer like AllReduceLinear to replace the existing nn.Linear layer.
        reversed (bool): this attribute is used to indicate whether tensors are reversed or not. Some models such as
            GPT1 and GPT2 use the transformers.modeling_utils.Conv1D layer instead of the nn.Linear layer.
            These layers store weight and bias tensors reversed.
        ignore_checker (bool): this attribute is used when you want to ignore errors in case the layers do not exist.
            Some models like Bert, Roberta and Electra have only encoder layers. but for Huggingface,
            these models are also designed to be able to used as decoders. In these models,
            some layers may or may not be created depending on the configuraions.
            In this case, you can use ignore_checker option to ignore errors even if the layers do not always exist.
    """

    weight: str = None
    bias: str = None
    n_fused: int = None
    replace: Any = None
    reversed: Any = None
    ignore_checker: bool = False


class Policy(ABC):
    """Policy object to apply parallelism per model"""

    def __init__(self, layer: nn.Module) -> None:
        """
        Constructor of Policy class
        Args:
            layer (nn.Module): The layer to apply the policy to
        """
        super().__init__()
        self.layer = layer

    @staticmethod
    def replace_arguments(config, world_size: int) -> Dict:
        """
        Policy for argument replacement for tp.

        Args:
            config (Config): Huggingface config object
            world_size (int): total number of gpu for parallelization

        Returns:
            Dict: Dictionary for argument replacement.
        Notes:
            The format of the dictionary object is as follows.
            dict:
                "param_1": reset_value_1,
                "param_2": reset_value_2,
                "param_3": reset_value_3,
                ...
                "param_n": reset_value_n
        """
        return {}

    @staticmethod
    def replace_modules() -> Dict:
        """
        Policy for class (module) replacement.

        Returns:
            Dict: Dictionary for class (module) replacement.

        Notes:
            The format of the dictionary object is as follows.

            dict:
                orig_class_name_1: reset_module_class_1,
                orig_class_name_2: reset_module_class_2,
                orig_class_name_3: reset_module_class_3,
                ...
                orig_class_name_n reset_module_class_n
        """
        return {}

    @staticmethod
    def attn_qkv() -> List:
        """
        Attention query, key, value projection layer

        Returns:
            List[Layer]: List of layer object
        """
        return []

    @staticmethod
    def attn_out() -> List:
        """
        Attention output projection layer

        Returns:
            List[Layer]: List of layer object
        """
        return []

    @staticmethod
    def mlp_in() -> List:
        """
        h -> 4h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return []

    @staticmethod
    def mlp_out() -> List:
        """
        4h -> h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return []

class AutoPolicy:

    def __init__(self):
        self.builtin_policies = {}

        with suppress(Exception):
            from transformers.models.gptj.modeling_gptj import (
                GPTJTrainedModel,
            )

            from policies.gpt_j import GPTJPolicy

            self.builtin_policies[GPTJPreTrainedModel] = [
                GPTJPolicy,
            ]


    def get_policy(self, model: nn.Module) -> Union[List[Policy], None]:
        """
        Find appropriate policies for the current model
        Args:
            model (nn.Module): model to parallelize
        Returns:
            Union[List[Policy], None]: appropriate policies or none
        """
        for k, v in self.available().items():
            if isinstance(model, k):
                return v
        return None

    def available(self):
        """Dictionary of available models and policies"""
        return self.builtin_policies