import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

import response_links
from vocabulary_links import build_index, lookup


class ResponseLinkTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parent / f".test-response-index-{uuid4().hex}"
        directory.mkdir()
        self.addCleanup(directory.rmdir)
        path = directory / "language.sqlite"
        self.addCleanup(path.unlink, missing_ok=True)
        records = [
            ("hello", "hello", "word"),
            ("odd]]", "odd]]", "token"),
            ("WANT TO GO", "WANT TO GO", "sequence"),
            ("WANT TO GO", "WANT TO GO", "sequence"),
            ("GO HOME NOW", "GO HOME NOW", "sequence"),
        ]
        build_index(path, iter(records))
        self.stats = response_links.build_response_index(path)
        self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.addCleanup(self.db.close)

    def test_continuation_for_a_combination_includes_exact_stored_evidence(self):
        result = response_links.respond(self.db, ["[[want]]", "[[to]]"])
        self.assertEqual(result["matched_context"], "want to")
        self.assertEqual(result["responses"], [
            {"link": "{@[[go]]}", "reply": "go", "evidence_count": 2, "example": "WANT TO GO"},
        ])
        self.assertEqual(lookup(self.db, "[[want to go]]")["occurrences"], 2)

    def test_long_input_backs_off_to_the_longest_evidenced_suffix(self):
        result = response_links.respond(self.db, ["[[hello]]", "[[go]]", "[[home]]"])
        self.assertEqual(result["matched_context"], "go home")
        self.assertEqual(result["responses"][0]["link"], "{@[[now]]}")

    def test_right_hand_phrase_can_be_composed_of_several_keys(self):
        result = response_links.respond(self.db, ["[[want]]"])
        self.assertIn("{@[[to]] [[go]]}", [entry["link"] for entry in result["responses"]])
        exact = response_links.respond(self.db, ["[[want]]"], reply="{@[[to]] [[go]]}")
        self.assertEqual(exact["evidence_count"], 2)

    def test_product_lookup_can_jump_to_the_last_page_of_ordered_responses(self):
        keys = [key for key, in self.db.execute("SELECT key FROM nodes ORDER BY id")]
        size = len(keys)
        result = response_links.respond(
            self.db, ["[[want]]", "[[to]]"], length=2, offset=size**2 - 2, limit=5,
        )
        self.assertEqual(result["responses_per_input"], str(size**2))
        self.assertEqual(result["ordered_pairs"], str(size**4))
        self.assertEqual([entry["position"] for entry in result["responses"]],
                         [str(size**2 - 2), str(size**2 - 1)])
        self.assertEqual(result["responses"][-1]["link"], "{@[[" + keys[-1] + "]] [[" + keys[-1] + "]]}")

    def test_reply_link_round_trips_brackets_without_evidence(self):
        odd_id = self.db.execute("SELECT id FROM nodes WHERE key = 'odd]]'").fetchone()[0]
        page = response_links.respond(self.db, ["[[hello]]"], length=1, offset=odd_id - 1, limit=1)
        link = page["responses"][0]["link"]
        result = response_links.respond(self.db, ["[[hello]]"], reply=link)
        self.assertEqual(result["link"], link)
        self.assertEqual(result["reply"], "odd]]")
        self.assertEqual(result["evidence_count"], 0)

    def test_decimal_offset_can_address_a_page_beyond_the_int_string_limit(self):
        self.assertEqual(response_links.decimal_position("1" + "0" * 5000), 10**5000)


if __name__ == "__main__":
    unittest.main()
