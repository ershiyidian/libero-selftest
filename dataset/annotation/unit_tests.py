import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from client import Provider, TeacherClient, parse_object, response_text
from events import DEFAULT_RELEASE, episode_events, read_selection
from run import build_packet, english_only, language_issues, validate_semantics
from prompts import ANNOTATION_LANGUAGE
from sft_dataset import AnnotatedEpisode


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = DEFAULT_RELEASE
        cls.rows = read_selection(cls.root)
        cls.by_key = {(row['task_id'], row['mode']): row for row in reversed(cls.rows)}

    def load(self, task, mode):
        row = self.by_key[(task, mode)]
        return row, episode_events(self.root, row)

    @staticmethod
    def semantic(claim, **extra):
        return {'claim': claim, 'slip_or_drop_claim': False, 'hidden_cause_claim': False,
                'entire_task_complete_claim': False, 'probe_subject_match': True,
                'action_match': True, 'natural': True,
                'issues': [], **extra}

    def test_final_release_contains_130_tasks_and_1040_unique_experts(self):
        self.assertEqual(len(self.rows), 1040)
        self.assertEqual(len({row['task_id'] for row in self.rows}), 130)
        self.assertEqual(len({(row['task_id'], row['demo']) for row in self.rows}), 1040)

    def test_all_events_are_causal_and_action_masked(self):
        count, ignored = 0, 0
        for row in self.rows:
            _, events, omitted = episode_events(self.root, row)
            mask = np.load(self.root / row['loss_mask_file'], allow_pickle=False)
            seen = set()
            for event in events:
                self.assertLessEqual(max(event.frame_steps), event.step)
                self.assertTrue(mask[event.step])
                self.assertNotIn(event.step, seen)
                seen.add(event.step)
                count += 1
            ignored += len(omitted)
        self.assertEqual(count, 2848)
        self.assertEqual(ignored, 248)

    def test_new_probe_return_boundary(self):
        _, (_, events, _) = self.load('L031', 'failed_recovery')
        feedback = [event for event in events if event.kind == 'feedback']
        self.assertEqual([(x.end, x.step) for x in feedback], [(45, 55), (91, 101)])
        self.assertEqual(feedback[1].oracle['outcome'], 'pass')

    def test_writer_packet_does_not_contain_physics_or_future_feedback(self):
        _, (meta, events, _) = self.load('L001', 'probe_success')
        event = next(event for event in events if event.kind == 'intent')
        packet = build_packet(meta, event, {}, [])
        self.assertIsNone(packet['probe_end'])
        text = json.dumps(packet, ensure_ascii=False)
        for forbidden in ('outcome', 'event_id', 'episode_id', 'probe_success', 'failed_recovery', 'stable_grasp_at_test_start', 'task_success_at_decision', 'friction'):
            self.assertNotIn(forbidden, text)
        self.assertLessEqual(max(packet['frame_steps']), packet['decision_step'])

    def test_push_correction_does_not_teach_lifting_plate(self):
        _, (_, events, _) = self.load('L026', 'failed_recovery')
        event = next(x for x in events if x.kind == 'feedback')
        self.assertIn('edge contact', event.planned_motion)
        self.assertNotIn('close the gripper', event.planned_motion)

    def test_multistep_task_does_not_change_current_probe_subject(self):
        _, (_, events, _) = self.load('L039', 'failed_recovery')
        event = next(x for x in events if x.kind == 'feedback')
        self.assertEqual(event.probe_focus, 'grasped_object')
        result = validate_semantics(self.semantic('negative', probe_subject_match=False), event,
                                    {'text': "The drawer didn't move, I'll try its handle again."})
        self.assertEqual(result['verdict'], 'rewrite')

    def test_semantic_boolean_fields_cannot_be_strings(self):
        _, (_, events, _) = self.load('L001', 'probe_success')
        with self.assertRaises(ValueError):
            validate_semantics(self.semantic('plan_only', natural='true'), events[0], {'text': "I'll try a gentle lift."})

    def test_textual_denial_of_actual_response_is_rejected(self):
        _, (_, events, _) = self.load('L031', 'failed_recovery')
        event = [x for x in events if x.kind == 'feedback'][-1]
        result = validate_semantics(self.semantic('negative'), event, {'text': "The drawer still hasn't moved."})
        self.assertEqual(result['verdict'], 'rewrite')

    def test_uncertainty_is_allowed_when_images_are_ambiguous(self):
        _, (_, events, _) = self.load('L031', 'failed_recovery')
        event = [x for x in events if x.kind == 'feedback'][-1]
        result = validate_semantics(self.semantic('uncertain'), event, {'text': "I can't tell if the drawer moved. I'll push gently again."})
        self.assertEqual(result['verdict'], 'accept')

    def test_failed_grasp_is_not_automatically_a_drop(self):
        _, (_, events, _) = self.load('L095', 'failed_recovery')
        event = next(x for x in events if x.kind == 'feedback')
        self.assertFalse(event.oracle['stable_grasp_at_test_start'])
        result = validate_semantics(self.semantic('negative', slip_or_drop_claim=True), event,
                                    {'text': "The bowl slipped out of my grip."})
        self.assertEqual(result['verdict'], 'rewrite')

    def test_probe_success_does_not_authorize_task_completion_claim(self):
        _, (_, events, _) = self.load('L001', 'probe_success')
        event = next(x for x in events if x.kind == 'feedback')
        self.assertFalse(event.oracle['task_success_at_decision'])
        result = validate_semantics(self.semantic('positive', entire_task_complete_claim=True), event,
                                    {'text': "The bowl is already placed on its target."})
        self.assertEqual(result['verdict'], 'rewrite')

    def test_invalid_evidence_and_annotation_voice_are_rejected(self):
        _, (_, events, _) = self.load('L001', 'probe_success')
        event = events[0]
        result = language_issues({'status': 'ok', 'text': '继续下一段示范。',
                                 'evidence_steps': [event.step + 1], 'observation': 'test',
                                 'next_action': 'test'}, event)
        self.assertGreaterEqual(len(result), 2)

    def test_english_only_label_and_auxiliary_fields(self):
        event = self.load('L001', 'probe_success')[1][1][0]
        data = {'status': 'ok', 'text': "I'll lift the bowl a little and check my grasp.",
                'observation': 'The gripper is close to the bowl rim.',
                'next_action': 'Try a small upward movement.',
                'evidence_steps': [event.frame_steps[-1]], 'confidence': 'clear'}
        self.assertEqual(language_issues(data, event), [])
        self.assertFalse(english_only('碗没有带起来。'))
        self.assertTrue(english_only(data['text']))
        self.assertIn('English', str(language_issues({**data, 'text': '碗没有带起来。'}, event)))
        self.assertIn('English', str(language_issues({**data, 'observation': '看不见。'}, event)))

    def test_training_input_excludes_current_language_and_keeps_action_masks(self):
        row, (meta, events, _) = self.load('L001', 'failed_recovery')
        first, second = events[:2]
        tokens = list(range(24))
        sidecar = {'trajectory_file': row['trajectory_file'], 'loss_mask_file': row['loss_mask_file'],
                   'annotation_language': ANNOTATION_LANGUAGE,
                   'task_language': meta['task_language'], 'events': [
                       {'step': first.step, 'text': "It didn't lift. I'll readjust my grip.", 'token_ids': tokens},
                       {'step': second.step, 'text': "I'll try a small lift again.", 'token_ids': tokens},
                   ]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'language.json'
            path.write_text(json.dumps(sidecar, ensure_ascii=False))
            dataset = AnnotatedEpisode(self.root, path)
            sample = dataset.read(second.step)
            self.assertEqual(sample['previous_text'], "It didn't lift. I'll readjust my grip.")
            self.assertNotIn("I'll try a small lift again.", sample['previous_text'])
            self.assertEqual(sample['language_text'], "I'll try a small lift again.")
            self.assertEqual(sample['actions'].shape, (8, 7))
            self.assertEqual(sample['history'].shape, (8, 7))
            original = np.load(self.root / row['loss_mask_file'], allow_pickle=False)
            np.testing.assert_array_equal(sample['action_mask'], original[second.step:second.step + 8])
            with self.assertRaisesRegex(ValueError, '禁止截断'):
                dataset.read(second.step, max_language_tokens=8)
            ordinary = dataset.read(first.step + 1)
            self.assertFalse(ordinary['language_supervised'])
            self.assertEqual(ordinary['previous_text'], "It didn't lift. I'll readjust my grip.")

    def test_return_phase_preserves_visual_response_evidence(self):
        row, (meta, events, _) = self.load('L031', 'failed_recovery')
        event = [x for x in events if x.kind == 'feedback'][-1]
        sidecar = {'trajectory_file': row['trajectory_file'], 'loss_mask_file': row['loss_mask_file'],
                   'annotation_language': ANNOTATION_LANGUAGE,
                   'task_language': meta['task_language'], 'events': [
                       {'step': event.step, 'text': "It moved slightly, I'll keep pushing.",
                        'token_ids': [1, 2], 'evidence_steps': event.frame_steps}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'language.json'
            path.write_text(json.dumps(sidecar, ensure_ascii=False))
            sample = AnnotatedEpisode(self.root, path).read(event.step)
            self.assertIn(event.end, sample['language_evidence_steps'])
            self.assertEqual(len(sample['language_evidence_images']), 2 * len(event.frame_steps))
            self.assertTrue(sample['requires_more_than_last_8_steps'])
            self.assertLessEqual(max(sample['language_evidence_steps']), event.step)

    def test_provider_limits_and_secrets(self):
        provider = Provider('buoke', 'example', 'https://example.invalid', 'SECRET_SHOULD_NOT_APPEAR', 6)
        self.assertNotIn('SECRET_SHOULD_NOT_APPEAR', repr(provider))
        self.assertNotIn('api_key', provider.public())
        with self.assertRaises(ValueError):
            TeacherClient(provider, concurrency=7)

    def test_response_json_parser(self):
        self.assertEqual(parse_object("""```json
{"text":"I'll lift it slightly."}
```""")['text'], "I'll lift it slightly.")
        text = response_text({'output': [{'type': 'reasoning', 'summary': []},
                                        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'ok'}]}]})
        self.assertEqual(text, 'ok')
        with self.assertRaises(ValueError):
            parse_object('[1,2]')


if __name__ == '__main__':
    unittest.main(verbosity=2)
