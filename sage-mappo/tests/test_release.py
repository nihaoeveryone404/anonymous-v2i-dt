from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msage_mappo.envs.v2x import VehicleToBSEnv
from msage_mappo.evaluation.report import curve_points, summarize, training_rows
from msage_mappo.llm_writer import QwenMemoryWriter
from msage_mappo.memory import SemanticMemoryBank
from msage_mappo.models.value_decomposition import QMixer
from msage_mappo.rewards.v2x import v2x_reward_from_info
from msage_mappo.training import baselines, full, safe_baseline, toy
from msage_mappo.training.matched_ablation import group_configs
from msage_mappo.utils.config import load_config, parse_config_args


class ReleaseTests(unittest.TestCase):
    def test_all_run_presets_parse(self):
        mapping = {
            "v2x_main": full, "v2x_smoke": full, "vmas_main": full,
            "v2x_ablation": full, "v2x_baselines": baselines,
            "v2x_baselines_smoke": baselines, "toy_smoke": toy,
            "v2x_safe_baseline": safe_baseline,
        }
        for name, runner in mapping.items():
            with self.subTest(config=name):
                args = runner.parse_args(["--config", f"configs/{name}.yaml"])
                self.assertGreater(args.episodes, 0)

    def test_preserved_source_definitions(self):
        manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
        for row in manifest["preserved_symbols"]:
            with self.subTest(symbol=row["symbol"]):
                tree = ast.parse((ROOT / row["target"]).read_text(encoding="utf-8"))
                node = next(n for n in tree.body if getattr(n, "name", None) == row["symbol"])
                actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual, row["ast_sha256"])

    def test_config_inheritance_and_cli_override(self):
        args = full.parse_args(["--config", "configs/v2x_smoke.yaml", "--episodes", "2"])
        self.assertEqual(args.episodes, 2)
        self.assertEqual(args.seeds, [7])
        self.assertFalse(args.v2x_align_neural_start)
        self.assertTrue(Path(args.v2x_env).is_file())
        self.assertEqual(args.v2x_power_weight, 2.2)
        aligned = full.parse_args(["--config", "configs/v2x_smoke.yaml", "--v2x-align-neural-start"])
        self.assertTrue(aligned.v2x_align_neural_start)

    def test_unknown_or_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "test.yaml"
            for content in ("episodez: 5", "episodes: bad", "v2x_align_neural_start: 'false'"):
                config.write_text(content, encoding="utf-8")
                with self.subTest(content=content), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    full.parse_args(["--config", str(config)])

    def test_config_cycle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "self.yaml"
            path.write_text("extends: self.yaml", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Circular"):
                load_config(path)

    def test_main_and_baseline_resource_settings_match(self):
        main = full.parse_args(["--config", "configs/v2x_main.yaml"])
        baseline = baselines.parse_args(["--config", "configs/v2x_baselines.yaml"])
        common = load_config(ROOT / "configs/v2x_common.yaml")
        for key in common:
            self.assertEqual(getattr(main, key), getattr(baseline, key), key)

    def test_environment_repeatability_and_packet_budget(self):
        with redirect_stdout(io.StringIO()):
            env1, env2 = VehicleToBSEnv(seed=7), VehicleToBSEnv(seed=7)
        first1, first2 = env1.reset(), env2.reset()
        np.testing.assert_array_equal(first1[0], first2[0])
        np.testing.assert_array_equal(first1[1], first2[1])
        for packet in ([0, 0, 0], [10, 0, 0], [2, 5, 9]):
            self.assertEqual(int(env1.project_packet_counts(packet).sum()), 10)
        cont = np.full((8, 3), 1 / 3, dtype=np.float32)
        disc = np.tile([4, 3, 3], (8, 1))
        r1, r2 = env1.step(cont, disc), env2.step(cont, disc)
        self.assertEqual(r1[2], r2[2])
        self.assertTrue(np.isfinite(r1[4]["avg_vehicle_max_delay_ms"]))

    def test_resource_reward_uses_power_and_guidance(self):
        args = full.parse_args(["--config", "configs/v2x_smoke.yaml"])
        info = {"avg_vehicle_max_delay_ms": 80., "avg_peak_power_usage": .4, "avg_peak_packet_usage": 4.}
        reward, _ = v2x_reward_from_info(-80, info, None, args)
        bonus_reward, _ = v2x_reward_from_info(-80, info, None, args, llm_guidance_bonus=.3)
        lower_power, _ = v2x_reward_from_info(-80, dict(info, avg_peak_power_usage=.2), None, args)
        self.assertAlmostEqual(bonus_reward - reward, .3)
        self.assertGreater(lower_power, reward)
        self.assertNotAlmostEqual(reward, -40.)

    def test_template_memory_and_explicit_qwen_failure(self):
        writer = QwenMemoryWriter("missing.gguf", backend="template")
        item = writer.write_memory({"violation_rate": 0.1, "constraint_tags": ["deadline_risk"]})
        self.assertEqual(item.writer_backend, "template")
        bank = SemanticMemoryBank()
        bank.add(item)
        embedding, retrieved = bank.retrieve("deadline risk", {"tags": ["deadline_risk"]})
        self.assertEqual(embedding.shape, (1, 64))
        self.assertEqual(retrieved[0], item)
        with self.assertRaises(FileNotFoundError):
            QwenMemoryWriter("missing.gguf", backend="qwen")

    def test_qmix_monotonicity(self):
        torch.manual_seed(7)
        mixer = QMixer(8, 80, 16)
        q = torch.randn(2, 8, requires_grad=True)
        output = mixer(q, torch.randn(2, 80))
        output.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue((q.grad >= -1e-6).all())

    def test_matched_groups_keep_shared_args(self):
        args = full.parse_args(["--config", "configs/v2x_ablation.yaml"])
        groups = group_configs(args)
        self.assertEqual([g[0] for g in groups], ["qwen", "template"])
        a, b = groups[0][1], groups[1][1]
        for key in a:
            if key not in {"llm_backend", "v2x_methods", "output"}:
                self.assertEqual(a[key], b[key], key)
        with tempfile.TemporaryDirectory() as folder:
            for name, config in groups:
                path = Path(folder) / f"{name}.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                parsed = full.parse_args(["--config", str(path)])
                self.assertEqual(parsed.llm_backend, name)
                self.assertEqual(parsed.seeds, args.seeds)
        args.allow_template_fallback = True
        with self.assertRaises(ValueError):
            group_configs(args)

    def test_summary_excludes_diagnostic_and_weights_seeds_equally(self):
        frame = pd.DataFrame([
            {"domain": "v2x", "scenario": "test", "method": "test", "seed": seed, "episode": ep,
             "episode_reward": value, "aligned_start": aligned}
            for seed, ep, value, aligned in [(7, 0, -999., 1), (7, 1, 1., 0), (7, 2, 3., 0), (42, 1, 10., 0)]
        ])
        raw = training_rows(frame)
        per_seed, summary = summarize(raw, 2)
        self.assertEqual(len(raw), 3)
        self.assertEqual(summary.episode_reward_mean.iloc[0], 6.)
        self.assertEqual(summary.min_episodes_used.iloc[0], 1)
        points = curve_points(raw[raw.seed == 7], "episode_reward", 1)
        self.assertTrue(points["std"].isna().all())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            training_rows(pd.concat([raw, raw], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
