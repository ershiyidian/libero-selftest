import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "singlepass_vla"))
from runtime import EpisodeHistory


class Tokenizer:
    eos_token_id = 2000000

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens if token < 2000000)


class HistoryContractTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.history = EpisodeHistory()

    def tokens(self, text, ending=None):
        return [ord(char) for char in text] + ([] if ending is None else [ending])

    def test_budget_cut_drops_partial_sentence(self):
        text = self.history.update_judgment(
            self.tokenizer, self.tokens("The bowl is on the table. I'll lift it toward"),
        )
        self.assertEqual(text, "The bowl is on the table.")
        self.assertEqual(self.history.previous_text, text)

    def test_partial_first_sentence_does_not_pollute_history(self):
        text = self.history.update_judgment(self.tokenizer, self.tokens("I'll move the bowl toward"))
        self.assertEqual(text, "")
        self.assertEqual(self.history.previous_text, "Ready to begin.")

    def test_model_stop_token_completes_unpunctuated_text(self):
        stop_ids = [self.tokenizer.eos_token_id, 2000001]
        text = self.history.update_judgment(
            self.tokenizer, self.tokens("Ready to release", 2000001), stop_ids,
        )
        self.assertEqual(text, "Ready to release")

    def test_consecutive_duplicate_keeps_earlier_evidence(self):
        for text in ("The bowl is on the table.", "I'll align the gripper.", "I'll close the gripper."):
            self.history.update_judgment(self.tokenizer, self.tokens(text))
        before = self.history.previous_text
        self.history.update_judgment(self.tokenizer, self.tokens("I'll close the gripper."))
        self.assertEqual(self.history.previous_text, before)

    def test_text_after_stop_is_never_read(self):
        tokens = self.tokens("I'll release the bowl.", self.tokenizer.eos_token_id) + self.tokens("Old text")
        text = self.history.update_judgment(self.tokenizer, tokens)
        self.assertEqual(text, "I'll release the bowl.")

    def test_tensor_input_matches_stop_by_value(self):
        import torch
        tokens = torch.tensor(
            self.tokens("Ready to release", self.tokenizer.eos_token_id) + self.tokens("Old text"),
        )
        text = self.history.update_judgment(self.tokenizer, tokens)
        self.assertEqual(text, "Ready to release")
        self.assertEqual(self.history.previous_text, text)


if __name__ == "__main__":
    unittest.main()
