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

if __name__ == "__main__":
    unittest.main()
