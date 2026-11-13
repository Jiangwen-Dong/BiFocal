"""Focused checks for grouped RL rollouts and advantages."""

import unittest
from unittest import mock

from bifocal.rl import episode as E
from bifocal.rl import sequence as S
from bifocal.rl import train_o as T
from bifocal.rl import train_reader as R


def row(video, question, rollout, reward):
    return {
        "schema_version": "online-memory-episode-v1",
        "episode_id": "%s:q%d:r%d" % (video, question, rollout),
        "group_id": "%s:q%d" % (video, question),
        "rollout_index": rollout, "split": "train",
        "uid": "%s:q%d" % (video, question), "video_id": video,
        "question_number": question,
        "budget": {"calls": 20, "frames": 256, "minutes": 10,
                   "new_tokens": 512},
        "sampling": {"seed": 10 + rollout, "temperature": 0.8, "top_p": 1.0},
        "policy_versions": {"base_model": "base", "o": "adapter",
                            "reader": "base", "reviewer": "base"},
        "memory_root_ref": "/memory/%s/q%d/r%d" % (video, question, rollout),
        "memory_fork_from": None,
        "memory_protocol": {"global_memory_enabled": True},
        "memory_budget": {"entry_tokens": 512, "branch_tokens": 2048,
                          "global_tokens": 2048, "events_per_read": 1},
        "steps": [{"t": 0, "actor": "o", "action": {"verb": "submit"},
                   "adapter": "o", "action_text": "SUBMIT B",
                   "valid": True, "calls": 1, "frames": 0,
                   "memory_ops": [], "prompt_token_ids": [1],
                   "completion_token_ids": [2],
                   "completion_logprobs": [-0.1],
                   "completion_raw_logprobs": [-0.1],
                   "allowed_token_ids": [[2]],
                   "generation_constraint": "action_menu_prefix_v1",
                   "logprob_kind": "sampling_post_warper",
                   "raw_logprob_kind": "raw_pre_warper", "use_cache": False,
                   "seed": 10 + rollout, "temperature": 0.8, "top_p": 1.0,
                   "top_k": 0, "finish_reason": "stop", "repair": False,
                   "normalized": False, "recovered_action": False,
                   "recovered_turn_cut": False}],
        "outcome": {"terminal": "agent_submit", "answer": "B",
                    "answered": True, "complete": True,
                    "correct": reward > 0, "calls": 1, "frames": 0,
                    "fault": "", "protocol_fallback": False,
                    "harness_recovery": False, "infrastructure_valid": True},
        "training_only": {"gold": "B", "correct": reward > 0,
                          "reward_components": {"task": reward,
                                                "cost": 0.0,
                                                "total": reward},
                          "reward": reward,
                          "step_returns": [{"reward": reward}],
                          "memory_rewards": {"global_memory": 0.1},
                          "optimization_reward_components": {
                              "task": reward, "cost": 0.0,
                              "total": reward},
                          "optimization_reward": reward,
                          "quality": "clean", "training_ready": True,
                          "hybrid_ready": True},
    }


class RLTests(unittest.TestCase):
    def test_four_rollout_sequence_gets_group_relative_advantages(self):
        sequence = {"video": "video", "questions": [1, 2],
                    "common_questions": [1, 2], "rollouts": []}
        for rollout, reward in enumerate((0.0, 1.0, 2.0, 3.0)):
            episodes = [row("video", q, rollout, reward) for q in (1, 2)]
            sequence["rollouts"].append({
                "rollout_index": rollout, "seed": 10 + rollout,
                "memory_root": "/memory/video/r%d" % rollout,
                "kept_questions": [1, 2],
                "question_start_sha256": [
                    {"question_number": 1, "sha256": "same"},
                    {"question_number": 2, "sha256": "same"}],
                "question_episodes": episodes,
                "skipped_questions": [],
            })
        summary, rows = S.admit_sequence(sequence, min_common_questions=2)
        self.assertTrue(summary["admitted"], summary["problems"])
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(
            sum(row["training_only"]["advantage"] for row in rows), 0.0, places=10)
        self.assertTrue(T.validate_sequence_training_groups(
            {"video": rows}, min_common_questions=2))
        with self.assertRaisesRegex(ValueError, "fewer than 3"):
            T.validate_sequence_training_groups({"video": rows})

    def test_shuffled_reward_control_permutes_advantages_within_a_group(self):
        def groups():
            return {"g": [row("video", 1, r, reward) for r, reward in
                          enumerate((-1.0, 1.0, 1.0, -1.0))]}
        plain = groups()
        T.attach_group_advantages(plain, advantage_mode="task_cost")
        shuffled = groups()
        T.attach_group_advantages(shuffled, advantage_mode="task_cost",
                                  shuffle_seed=7)
        original = [r["training_only"]["advantage"] for r in plain["g"]]
        permuted = [r["training_only"]["advantage"] for r in shuffled["g"]]
        self.assertEqual(sorted(original), sorted(permuted))
        again = groups()
        T.attach_group_advantages(again, advantage_mode="task_cost",
                                  shuffle_seed=7)
        self.assertEqual(permuted,
                         [r["training_only"]["advantage"] for r in again["g"]])
        self.assertEqual(S.permute_within_group([1, 2, 3, 4], 7, "g").count(1), 1)

    def test_reader_prepare_enforces_the_configured_question_floor(self):
        sequence = {"video": "video", "questions": [1, 2],
                    "common_questions": [1, 2], "rollouts": []}
        for rollout, reward in enumerate((0.0, 1.0, 2.0, 3.0)):
            episodes = [row("video", q, rollout, reward) for q in (1, 2)]
            sequence["rollouts"].append({
                "rollout_index": rollout, "seed": 10 + rollout,
                "memory_root": "/memory/video/r%d" % rollout,
                "kept_questions": [1, 2],
                "question_start_sha256": [
                    {"question_number": 1, "sha256": "same"},
                    {"question_number": 2, "sha256": "same"}],
                "question_episodes": episodes,
                "skipped_questions": [],
            })
        _, rows = S.admit_sequence(sequence, min_common_questions=2)
        config = {"collection": {
            "group_size": 4, "expected_groups": 1,
            "sequence_level": True, "min_common_questions": 3,
        }}
        with mock.patch.object(R, "_load_groups", return_value={"video": rows}):
            with self.assertRaisesRegex(ValueError, "fewer than 3"):
                R.prepare("unused", config)

    def test_incomplete_episode_is_not_training_ready(self):
        sample = row("video", 1, 0, 1.0)
        sample["outcome"]["complete"] = False
        problems = E.validate_episode(sample)
        self.assertTrue(problems)

    def test_sequence_requires_four_rollouts(self):
        with self.assertRaises(ValueError):
            S.attach_sequence_advantages([])


if __name__ == "__main__":
    unittest.main()
