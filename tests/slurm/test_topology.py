"""Tests for the parallelism topology gate."""

import unittest

from llmflux.slurm.topology import Topology, TopologyError, resolve


class TestResolveValidShapes(unittest.TestCase):
    def test_single_node_single_gpu(self):
        t = resolve(1, 1, "vllm")
        self.assertEqual((t.nodes, t.gpus_per_node), (1, 1))
        self.assertFalse(t.is_multi_node)

    def test_single_node_multi_gpu_is_allowed_for_both_engines(self):
        for engine in ("vllm", "ollama"):
            with self.subTest(engine=engine):
                t = resolve(1, 4, engine)
                self.assertEqual(t.gpus_per_node, 4)
                self.assertEqual(t.tensor_parallel_size, 4)
                self.assertEqual(t.world_size, 4)

    def test_string_inputs_are_coerced(self):
        """SlurmConfig values arrive as ints, but env-sourced values are strings."""
        t = resolve("1", "2", "vllm")
        self.assertEqual((t.nodes, t.gpus_per_node), (1, 2))

    def test_topology_is_frozen(self):
        t = resolve(1, 2, "vllm")
        with self.assertRaises(Exception):
            t.nodes = 4


class TestResolveMultiNode(unittest.TestCase):
    def test_vllm_multi_node_is_allowed(self):
        t = resolve(2, 4, "vllm")
        self.assertTrue(t.is_multi_node)
        self.assertEqual(t.world_size, 8)

    def test_parallelism_axes_map_to_the_allocation(self):
        """TP within a node, PP across them — the mapping the launcher relies on."""
        t = resolve(2, 4, "vllm")
        self.assertEqual(t.tensor_parallel_size, 4, "TP is per node")
        self.assertEqual(t.pipeline_parallel_size, 2, "PP is one stage per node")
        self.assertEqual(
            t.tensor_parallel_size * t.pipeline_parallel_size, t.world_size,
            "TP x PP must account for every allocated GPU",
        )

    def test_ollama_multi_node_is_still_rejected(self):
        """Ollama has no distributed-inference story; nodes > 1 would idle them."""
        with self.assertRaises(TopologyError):
            resolve(2, 4, "ollama")

    def test_message_is_actionable(self):
        """The message has to tell the user what to do, not just say no."""
        with self.assertRaises(TopologyError) as ctx:
            resolve(4, 2, "ollama")
        msg = str(ctx.exception)
        self.assertIn("--nodes 4", msg)
        self.assertIn("--gpus-per-node", msg, "must name the flag to use instead")
        self.assertIn("idle", msg, "must say what the extra nodes would do")
        self.assertIn("issues/137", msg, "must point at the tracking issue")

    def test_message_reports_the_wasted_node_count(self):
        with self.assertRaises(TopologyError) as ctx:
            resolve(4, 2, "ollama")
        self.assertIn("3 node(s)", str(ctx.exception))


class TestResolveRejectsNonsense(unittest.TestCase):
    def test_zero_or_negative_nodes(self):
        for n in (0, -1):
            with self.subTest(nodes=n):
                with self.assertRaises(TopologyError):
                    resolve(n, 1, "vllm")

    def test_zero_or_negative_gpus(self):
        for g in (0, -1):
            with self.subTest(gpus=g):
                with self.assertRaises(TopologyError):
                    resolve(1, g, "vllm")

    def test_non_integer_values(self):
        with self.assertRaises(TopologyError):
            resolve("many", 1, "vllm")


class TestTopologyProperties(unittest.TestCase):
    def test_world_size_is_nodes_times_gpus(self):
        t = Topology(nodes=2, gpus_per_node=4, engine="vllm")
        self.assertEqual(t.world_size, 8)

    def test_tensor_parallel_is_within_node(self):
        """TP is per node; it must not silently become the world size."""
        t = Topology(nodes=2, gpus_per_node=4, engine="vllm")
        self.assertEqual(t.tensor_parallel_size, 4)



class TestRunnerGatesAtSubmitTime(unittest.TestCase):
    """The gate must fire from the runner, before a job is ever queued.

    Validating in `topology.resolve` is worthless if a caller can reach `sbatch`
    without going through it. Both runner entry points are covered: `run()`
    (which `benchmark` also uses) and `serve()`, which has its own code path.
    """

    def _runner(self, nodes, engine="ollama"):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from llmflux.core.config import Config, SlurmConfig
        from llmflux.slurm.runner import SlurmRunner

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for sub in ("data", "models", "logs", "containers"):
            (root / sub).mkdir()

        cfg = Config(
            data_dir=str(root / "data"),
            models_dir=str(root / "models"),
            logs_dir=str(root / "logs"),
            containers_dir=str(root / "containers"),
            slurm=SlurmConfig(
                partition="gpu", nodes=nodes, gpus_per_node=2,
                time="01:00:00", memory="16G", cpus_per_task=4, account="acct",
            ),
        )
        patcher = patch("llmflux.slurm.runner.ConfigManager")
        mgr = patcher.start()
        self.addCleanup(patcher.stop)
        mgr.return_value.get_config.return_value = cfg

        from llmflux.core.config import EngineConfig
        return SlurmRunner(config=cfg.slurm, engine_config=EngineConfig(engine=engine))

    def test_run_rejects_multi_node_without_submitting(self):
        from unittest.mock import patch
        runner = self._runner(nodes=2)
        with patch("subprocess.run") as sub:
            with self.assertRaises(TopologyError):
                runner.run(input_path="/nonexistent.jsonl")
            sub.assert_not_called()

    def test_serve_rejects_multi_node_without_submitting(self):
        from unittest.mock import patch
        runner = self._runner(nodes=2)
        with patch("subprocess.run") as sub:
            with self.assertRaises(TopologyError):
                runner.serve(email="x@example.edu")
            sub.assert_not_called()

    def test_single_node_still_reaches_past_the_gate(self):
        """The gate must not become a blanket refusal for everyone."""
        runner = self._runner(nodes=1, engine="vllm")
        try:
            runner.run(input_path="/nonexistent.jsonl")
        except TopologyError:
            self.fail("nodes=1 must not be rejected by the topology gate")
        except Exception:
            pass  # any later failure is fine; we only care about the gate


class TestParallelismMustFillTheAllocation(unittest.TestCase):
    """User-supplied engine args beat derived ones — including into a shape the
    allocation cannot serve.

    A value the user typed is never silently overridden, which means the user can
    contradict their own allocation. At nodes == 1 that is their business. At
    nodes > 1 it is the silent-waste bug the launcher exists to remove, and
    nothing downstream catches it: vLLM asserts only `world_size % nnodes == 0`,
    so tp=4 pp=1 on a 2-node 8-GPU allocation passes and runs on half the job.
    """

    def _args(self, nodes, gpus_per_node, engine_args=None):
        from unittest.mock import MagicMock
        from llmflux.slurm.runner import SlurmRunner

        runner = SlurmRunner.__new__(SlurmRunner)
        runner.slurm_config = MagicMock(nodes=nodes, gpus_per_node=gpus_per_node)
        runner.engine = MagicMock(engine="vllm")
        return runner._resolve_vllm_engine_args({"vllm_engine_args": engine_args})

    def test_derived_parallelism_fills_the_allocation(self):
        """The line that makes the second node participate.

        Pipeline parallelism is the axis that spans nodes; without this flag the
        extra nodes are allocated and idle, which is the whole bug. Nothing else
        in the suite asserts it reaches the engine args.
        """
        args = self._args(2, 4)
        self.assertIn("--tensor-parallel-size 4", args)
        self.assertIn("--pipeline-parallel-size 2", args)

    def test_four_nodes(self):
        args = self._args(4, 4)
        self.assertIn("--pipeline-parallel-size 4", args)

    def test_single_node_gets_no_pipeline_stage(self):
        args = self._args(1, 4)
        self.assertIn("--tensor-parallel-size 4", args)
        self.assertNotIn("pipeline-parallel-size", args)

    def test_conflicting_value_is_rejected(self):
        """Both directions, and every spelling that reaches the same vLLM flag.

        vLLM's FlexibleArgumentParser normalises "_" to "-" and argparse takes
        the last occurrence, so an unfolded spelling would emit two copies of one
        flag and the derived value would silently beat the user's.
        """
        cases = [
            ('{"pipeline-parallel-size": 1}', "uses 4 GPU"),
            ('{"tensor-parallel-size": 2}', "uses 4 GPU"),
            ('{"tensor-parallel-size": 8}', "uses 16 GPU"),
            ('{"pipeline_parallel_size": 1}', "uses 4 GPU"),
            ('{"--pipeline-parallel-size": 1}', "uses 4 GPU"),
            ('{"--tensor_parallel_size": 8}', "uses 16 GPU"),
        ]
        for engine_args, expected in cases:
            with self.subTest(engine_args=engine_args):
                with self.assertRaises(TopologyError) as ctx:
                    self._args(2, 4, engine_args)
                msg = str(ctx.exception)
                self.assertIn(expected, msg)
                self.assertIn("8 GPU(s)", msg, "must name the allocation")
                self.assertIn("--nodes 2", msg, "must name the shape the user asked for")

    def test_a_matching_override_is_allowed(self):
        """Rejecting every override would be a blanket refusal, not a gate.

        tp=8 pp=1 across 2 nodes is unusual — TP spanning nodes — but it does
        account for all 8 GPUs, so it is the user's call.
        """
        args = self._args(2, 4, '{"tensor-parallel-size": 8, "pipeline-parallel-size": 1}')
        self.assertIn("--tensor-parallel-size 8", args)
        self.assertIn("--pipeline-parallel-size 1", args)

    def test_single_node_override_is_untouched(self):
        """At nodes == 1 asking for fewer GPUs than allocated is legitimate."""
        args = self._args(1, 4, '{"tensor-parallel-size": 2}')
        self.assertEqual(args, "--tensor-parallel-size 2")

    def test_unrelated_engine_args_survive_canonicalisation(self):
        args = self._args(1, 1, '{"max-model-len": 8192, "trust-remote-code": true}')
        self.assertIn("--max-model-len 8192", args)
        self.assertIn("--trust-remote-code", args)

    def test_a_non_integer_parallelism_is_reported_not_crashed(self):
        with self.assertRaises(TopologyError):
            self._args(2, 4, '{"tensor-parallel-size": "four"}')


if __name__ == "__main__":
    unittest.main()
