"""Focused checks for BiFocal's local and global memory contract."""

import unittest

from bifocal.memory import local as B


ROLES = {
    "R1": "reader", "R2": "reader",
    "O1": "orchestrator", "H1": "harness",
}


def claim(value="2", unit="minute", predicate="time"):
    return {"subject": "The Woman", "predicate": predicate,
            "value": value, "unit": unit}


def leaf(local_id="L1", question_number=1, reader="R1"):
    return B.make_local_leaf(
        local_id=local_id, video="video-1", question_number=question_number,
        reader=reader, chunk=1, turn=1, read_attempt=1,
        assigned_span=[0.0, 64.0], sighting_span=[10.0, 12.0],
        chunk_span=[0.0, 64.0], description="the woman waits for two minutes",
        frame_handle="frames/L1", stored_frames=32, raw_claim=claim(),
        provisional=False, version=1, parent_local_id=None,
    )


def reviewed():
    item = leaf()
    return B.review_local(
        item, reviewer="O1", actor_roles=ROLES, status="accepted",
        frames_from_store=True, reason="checked stored frames",
        review_call_id="review-1",
        review_call_log_entries=[{"call_id": "review-1"}],
        reviewed_frame_handles=[item["frame_handle"]],
        prompt_tokens=20, stored_frames_read=32, answer_support=True,
        reader_answer="B", reader_quote="the woman waits for two minutes",
    )


class MemoryTests(unittest.TestCase):
    def test_claim_units_are_normalized(self):
        minutes = B.normalize_claim("video-1", "woman", "time", "2", "minute")
        seconds = B.normalize_claim("video-1", "woman", "time", "120", "second")
        self.assertEqual(minutes["key"], seconds["key"])
        self.assertEqual(minutes["value"], "120")

    def test_reviewed_leaf_promotes_to_a_citable_global_record(self):
        item = reviewed()
        records = B.promote_local([], item, 2, "O1", ROLES)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["citation_allowed"])
        self.assertTrue(B.is_citable(records[0]))

    def test_author_cannot_review_their_own_leaf(self):
        with self.assertRaisesRegex(ValueError, "different actor"):
            B.review_local(
                leaf(reader="O1"), reviewer="O1", actor_roles=ROLES,
                status="accepted", frames_from_store=True, reason="self review",
                review_call_id="review-1",
                review_call_log_entries=[{"call_id": "review-1"}],
                reviewed_frame_handles=["frames/L1"], prompt_tokens=20,
                stored_frames_read=32,
            )

    def test_acceptance_requires_stored_frames(self):
        with self.assertRaisesRegex(ValueError, "stored frames"):
            B.review_local(
                leaf(), reviewer="O1", actor_roles=ROLES, status="accepted",
                frames_from_store=False, reason="looked away",
                review_call_id="review-1",
                review_call_log_entries=[{"call_id": "review-1"}],
                reviewed_frame_handles=[], prompt_tokens=20,
                stored_frames_read=0,
            )


if __name__ == "__main__":
    unittest.main()
