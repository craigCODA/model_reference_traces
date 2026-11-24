import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

from vocabulary_links import build_index, combine, lookup


class VocabularyLinkTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f".test-language-index-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.root.rmdir)
        path = self.root / "language.sqlite"
        self.addCleanup(path.unlink, missing_ok=True)
        self.records = [
            ("the", "the", "token"),
            ("Ġapple", " apple", "token"),
            ("apple", "apple", "word"),
            ("apple", "apple", "word"),
            ("fell", "fell", "word"),
            ("THE APPLE FELL", "THE APPLE FELL", "sequence"),
            ("APPLE FELL", "APPLE FELL", "sequence"),
            ("the apple", "the apple", "sequence"),
            ("apple apple fell", "apple apple fell", "sequence"),
            ("苹果", "苹果", "word"),
            ("果", "果", "character"),
        ]
        self.stats = build_index(path, iter(self.records))
        self.db = sqlite3.connect(path)
        self.addCleanup(self.db.close)

    def test_literal_variants_and_duplicate_rows_survive_one_shared_node(self):
        view = lookup(self.db, "[[Ġapple]]")
        self.assertEqual(view["key"], "apple")
        self.assertEqual(view["occurrences"], 3)
        self.assertIn(("Ġapple", "token", 1), view["variants"])
        self.assertIn(("apple", "word", 2), view["variants"])
        self.assertEqual(self.stats["raw_entries"], len(self.records))

    def test_stored_sequences_link_to_terms_and_known_shorter_phrases(self):
        view = lookup(self.db, "[[THE APPLE FELL]]")
        self.assertEqual(set(view["members"]), {"the", "apple", "fell", "the apple", "apple fell"})
        self.assertIn("the apple fell", lookup(self.db, "[[the apple]]")["containers"])

    def test_combination_keeps_order_for_any_number_of_items(self):
        exact = combine(self.db, ["[[the]]", "[[apple]]", "[[fell]]"])
        self.assertEqual(exact["sequence"], "the apple fell")
        self.assertEqual(exact["stored_occurrences"], 1)
        long = combine(self.db, ["[[the]]", "[[apple]]"] * 10)
        self.assertEqual(long["links"], ["the", "apple"] * 10)
        self.assertEqual(long["stored_occurrences"], 0)
        self.assertEqual(long["shared_sequences"], [])

    def test_shared_sequence_uses_the_requested_number_of_occurrences(self):
        result = combine(self.db, ["[[apple]]", "[[apple]]"])
        self.assertEqual(result["shared_sequences"], ["apple apple fell"])

    def test_character_can_link_to_a_stored_word(self):
        self.assertIn("苹果", lookup(self.db, "[[果]]")["containers"])


if __name__ == "__main__":
    unittest.main()
