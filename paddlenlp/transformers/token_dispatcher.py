from abc import ABC, abstractmethod
import paddle
from paddle.distributed.communication.group import Group
from typing import List, Optional, Tuple
import paddle.distributed.fleet as fleet
import paddle.distributed as dist
from fused_a2a import fused_dispatch, fused_combine
from moe_utils import permute, unpermute

class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(self, routing_map: paddle.Tensor, probs: paddle.Tensor):
        """Set up metadata of routing_map and probs."""
        pass

    @abstractmethod
    def dispatch(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispached_metadata(self) -> paddle.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _DeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: float = None,
        num_experts: int = None,
        num_local_experts: int = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts

        # Metadata
        self.token_indices = None
        self.token_probs = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

    def setup_metadata(self, routing_map: paddle.Tensor, probs: paddle.Tensor):
        num_tokens = routing_map.shape[0]

        routing_map = routing_map.reshape([num_tokens, self.num_experts])
        probs = probs.reshape([num_tokens, self.num_experts])
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = paddle.topk(probs, self.router_topk, axis=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        # hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
        states = dict()
        hidden_states, dispatched_probs = (
            fused_dispatch(
                hidden_states, self.token_indices, self.token_probs, self.num_experts, states, self.group
            )
        )
        self.handle = states['handle']
        self.tokens_per_expert = states['tokens_per_expert']
        self.dispatched_indices = states['dispatched_indices']
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (paddle.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (paddle.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.int64
        )

        multihot_probs = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.float32
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = paddle.arange(batch_size).repeat_interleave(
            mask.sum(axis=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.cast(paddle.bool), multihot_probs

    def get_dispached_metadata(self) -> paddle.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> paddle.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        # hidden_states, event = fused_combine(hidden_states, self.group, self.handle)
        hidden_states = fused_combine(hidden_states, self.group, self.handle)
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
            self.dispatched_indices, self.dispatched_probs
        )
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            num_out_tokens=sum(self.tokens_per_expert),
            fused=self.permute_fusion,
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        assert self.dispatched_probs.dtype == paddle.float32, "DeepEP only supports float32 probs"
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.dispatched_probs,
            fused=self.permute_fusion,
        )
        return hidden_states.to(input_dtype)

class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.shared_experts: Optional[SharedExpertMLP] = None

        # self.tp_size = config.expert_tensor_parallel_size
        # self.ep_size = config.expert_model_parallel_size
        self.tp_size = 1
        self.ep_size = 8

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        # return get_expert_model_parallel_group()
        raise NotImplementedError("ep_group function not implemented.")

    @property
    def tp_group(self):
        """Get expert tensor parallel group."""
        # return get_expert_tensor_parallel_group()
        raise NotImplementedError("tp_group function not implemented.")

    @property
    def tp_rank(self):
        """Get expert tensor parallel rank."""
        # return get_expert_tensor_parallel_rank()
        raise NotImplementedError("tp_rank function not implemented.")

    @property
    def tp_ep_group(self):
        """Get expert tensor and model parallel group."""
        # return get_expert_tensor_and_model_parallel_group()
        # raise NotImplementedError("tp_ep_group function not implemented.")
        hcg = fleet.get_hybrid_communicate_group()
        etp_group = hcg.get_model_parallel_group()
        return etp_group

    @abstractmethod
    def token_permutation(
        self, tokens: paddle.Tensor, probs: paddle.Tensor, routing_map: paddle.Tensor
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (paddle.Tensor): Input tokens.
            probs (paddle.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (paddle.Tensor): Token to expert mapping tensor.

        Returns:
            paddle.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(self, expert_output: paddle.Tensor, bias: paddle.Tensor = None):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (paddle.Tensor): The output tensor from the expert models.
            bias (paddle.Tensor): The bias tensor.

        Returns:
            (paddle.Tensor, paddle.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        # assert self.config.moe_shared_expert_overlap
        # self.shared_experts = shared_experts
        raise NotImplementedError("set_shared_experts function not implemented.")

class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Flexible token dispatcher for MoE models with Efficient-A2A communication kernels.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], moe_router_topk, moe_expert_capacity_factor, num_moe_experts
    ):
        super().__init__()

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"
        self._comm_manager = _DeepepManager(
            group=self.tp_ep_group,
            router_topk=self.tp_size * moe_router_topk,
            permute_fusion=False,
            capacity_factor=moe_expert_capacity_factor,
            num_experts=self.tp_size * num_moe_experts,
            num_local_experts=self.num_local_experts,
        )

    def set_shared_experts(self, shared_experts):
        raise NotImplementedError("Shared experts overlap not supported in flex token dispatcher")

    def _initialize_metadata(self, routing_map: paddle.Tensor, probs: paddle.Tensor) -> paddle.Tensor:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.

        This function expands the routing_map from shape [num_local_tokens, num_experts] to
        [num_local_tokens, world_size, num_local_experts]. Each element in the routing_map
        indicates whether a token should be sent to a specific rank. Specifically, the
        routing_map is replicated across TP group since each TP ranks in a TP group should
        receive the same tokens.
        """
        num_local_tokens = routing_map.shape[0]
        world_size = self.tp_size * self.ep_size
        # Organize routing map and probs to [num_local_tokens, world_size, num_local_experts]
        routing_map = (
            routing_map.reshape([num_local_tokens, self.ep_size, 1, self.num_local_experts])
            .expand([-1, -1, self.tp_size, -1])
            .reshape([num_local_tokens, world_size, self.num_local_experts])
        ).contiguous()
        probs = (
            probs.reshape([num_local_tokens, self.ep_size, 1, self.num_local_experts])
            .expand([-1, -1, self.tp_size, -1])
            .reshape([num_local_tokens, world_size, self.num_local_experts])
        ).contiguous()
        return routing_map, probs

    def token_permutation(
        self, hidden_states: paddle.Tensor, probs: paddle.Tensor, routing_map: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])

        # Initialize metadata
        routing_map, probs = self._initialize_metadata(routing_map, probs)

        self._comm_manager.setup_metadata(routing_map, probs)
        hidden_states = self._comm_manager.dispatch(hidden_states)
        global_input_tokens = self._comm_manager.get_permuted_hidden_states_by_experts(
            hidden_states
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: paddle.Tensor, bias: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        assert bias is None, "Bias is not supported in MoEFlexTokenDispatcher"
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        hidden_states = self._comm_manager.combine(hidden_states)

        return hidden_states.view(self.hidden_shape), None
