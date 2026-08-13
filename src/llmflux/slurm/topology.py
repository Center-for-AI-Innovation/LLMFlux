"""Parallelism topology: how a job's nodes and GPUs map onto engine parallelism.

Single place that decides whether a requested SLURM shape can actually be
served, and how it maps onto engine parallelism. It exists because the mapping
was previously implicit and unvalidated: `nodes` was accepted, forwarded to
`#SBATCH --nodes=`, and then ignored by everything else, so a multi-node request
allocated nodes that no process ever used.

Validation is deliberately a gate that *narrows* as capability arrives, rather
than a permanent refusal. Today `nodes > 1` is rejected for every engine. As
multi-node lands, the rejection narrows to the cases that genuinely cannot be
served (Ollama, model architectures without pipeline-parallel support) instead
of being deleted.
"""

from dataclasses import dataclass

#: Engines that can use more than one node. Empty until the launcher lands.
_MULTINODE_CAPABLE_ENGINES = frozenset()

#: Tracking issue for multi-node support.
_TRACKING_ISSUE = "https://github.com/Center-for-AI-Innovation/LLMFlux/issues/137"


class TopologyError(ValueError):
    """A requested node/GPU shape cannot be served.

    Raised at submit time, before a job is queued, so the user finds out in
    milliseconds rather than after a queue wait.
    """


@dataclass(frozen=True)
class Topology:
    """A validated mapping from a SLURM allocation to engine parallelism."""

    nodes: int
    gpus_per_node: int
    engine: str

    @property
    def world_size(self) -> int:
        """Total GPUs across the allocation."""
        return self.nodes * self.gpus_per_node

    @property
    def tensor_parallel_size(self) -> int:
        """GPUs the model is sharded across within one node."""
        return self.gpus_per_node

    @property
    def is_multi_node(self) -> bool:
        return self.nodes > 1


def resolve(nodes, gpus_per_node, engine) -> Topology:
    """Validate a requested shape and return the topology it implies.

    Args:
        nodes: nodes requested (``SlurmConfig.nodes``)
        gpus_per_node: GPUs per node requested (``SlurmConfig.gpus_per_node``)
        engine: engine name, e.g. ``"vllm"`` or ``"ollama"``

    Returns:
        The validated :class:`Topology`.

    Raises:
        TopologyError: if the shape cannot be served.
    """
    try:
        nodes = int(nodes)
        gpus_per_node = int(gpus_per_node)
    except (TypeError, ValueError) as exc:
        raise TopologyError(
            f"nodes and gpus_per_node must be integers, got "
            f"nodes={nodes!r}, gpus_per_node={gpus_per_node!r}"
        ) from exc

    if nodes < 1:
        raise TopologyError(f"nodes must be at least 1, got {nodes}")
    if gpus_per_node < 1:
        raise TopologyError(
            f"gpus_per_node must be at least 1, got {gpus_per_node}"
        )

    if nodes > 1 and engine not in _MULTINODE_CAPABLE_ENGINES:
        raise TopologyError(
            f"--nodes {nodes} was requested, but multi-node inference is not "
            f"implemented"
            + (f" for the {engine} engine" if engine else "")
            + ".\n"
            f"The extra {nodes - 1} node(s) would be allocated and billed for "
            f"the job's full walltime while sitting completely idle: the "
            f"inference server only ever starts on the first node.\n"
            f"Use --nodes 1 and scale with --gpus-per-node instead "
            f"(currently {gpus_per_node}).\n"
            f"Multi-node support is tracked at {_TRACKING_ISSUE}"
        )

    return Topology(nodes=nodes, gpus_per_node=gpus_per_node, engine=engine)
