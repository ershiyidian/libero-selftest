import json


PROMPT_VERSION = 'en-v1-causal-visual-physics'
ANNOTATION_LANGUAGE = 'en'

WRITER_RULES = """You are annotating an embodied robot's demonstration with an occasional, natural first-person English decision.
The English sentence is a training target for the robot's pretrained language model. The original task instruction is already in English. Produce ALL language fields (text, observation, next_action, skip_reason) in English. Never output Chinese or a translation.

You see only the robot's current and earlier images and its past proprioceptive motion. Each image strip shows the fixed camera on the left, wrist camera on the right, arranged from earlier to later. Wrist-camera motion may reflect camera motion, not object motion. Compare the fixed view.

At an INTENT event, describe a specific small test you are ABOUT to perform. You have not observed the test result yet: do not say whether it will work. Natural examples of STYLE ONLY (not answers): 'I'll lift it slightly to check the grasp.' / 'Let me try a gentle push and see if it moves.'
At a FEEDBACK event, compare TEST START with TEST RESPONSE to decide what changed and what to do next. AFTER RETURN is the state after the hand retracts; returning close to the initial pose does not mean the test caused no motion. If response is too subtle or occluded, name the exact uncertainty. Examples of STYLE ONLY: 'It stayed on the table. I'll adjust my grip.' / 'The drawer moved a little. I'll keep pushing.'

The response must accurately concern probe_focus: grasped_object means the held object, drawer the drawer, door the door, switch the control knob/switch, pushed_object the pushed object. A multi-object task might include both a drawer and a bowl; describe the object currently being tested.

Only infer effects actually visible up to decision_step. Do not assert friction, material, weight, grip force, hidden contact forces, exact failure mechanisms, or the success of the whole task. A closed gripper does not guarantee it holds anything. Failure to lift is not automatically slipping or dropping. Do not describe a door crack already present before the test as newly opened. For subtle displacements, do not exaggerate how far it moved.

Use the next planned action ONLY to say what to do next; it is not evidence that this future action already happened. planned_control_mean_xyz is a mean future position command; positive z means upward, negative z downward. It is fine to say 'I'll adjust my position' instead of naming low-level movements if the control intent is unclear.

Use simple, specific, natural English, usually one or two short clauses (roughly 8–22 words). Vary ordinary phrasing without filler or theatrical language. No technical narration: never say 'in the next demonstration', 'following the expert trajectory', 'the model predicts', or 'stage completed'. Avoid generic motivational phrases and repeated robotic templates.

Return ONE JSON object, with English string values:
{"status":"ok"|"skip","text":"short natural English thought","observation":"what current or previous images actually show","next_action":"immediate intended action","evidence_steps":[integer frame indices provided],"confidence":"clear"|"uncertain","skip_reason":"English reason only if skipped"}
"""

SEMANTIC_RULES = """Read the robot's English thought as language ONLY. You cannot see the images and you do not know the physical outcome. Classify only what this English text CLAIMS.
Return a single JSON object:
{"claim":"plan_only|positive|negative|uncertain|none","slip_or_drop_claim":false,"hidden_cause_claim":false,"entire_task_complete_claim":false,"probe_subject_match":true,"action_match":true,"natural":true,"issues":[]}

plan_only: tests or actions proposed in the future without claiming an outcome.
positive: claims that the TESTED object actually responded/moved/was carried or held.
negative: claims it did not respond, did not lift or hold, or the test failed.
uncertain: specifically says it cannot tell WHETHER the tested object moved/was held.
none: no physical observation and no meaningful intended test.

'It seems to have moved, but I am not sure it is open' is POSITIVE about motion. 'I cannot tell whether it moved' is UNCERTAIN. 'It slipped out of my gripper' claims a slip/drop; 'I could not lift it' does not.
Mark slip_or_drop_claim for explicit slip, fall, drop, loss of previously held object. Mark hidden_cause_claim for unsupported friction/weight/material/grip-force explanations. Mark entire_task_complete_claim only if the sentence claims completion, not if it proposes to finish.
probe_subject_match: feedback must concern probe_focus (grasped_object, drawer, door, switch, pushed_object) even if the overall task mentions other items.
action_match: compare the proposed next action with next_action_intent and next_control_mean_xyz. Moving back to align should not be described as carrying an object upward; sliding a plate should not be described as picking it up. Avoid over-interpreting a coarse generic next-action plan.
natural: well-formed conversational English without theatrical prose, meta-annotation, descriptions of datasets/trajectories, or a Chinese phrase.
Do not judge whether the actual test succeeded, do not hallucinate visual evidence, and do not return accept/reject.
"""


def writer_prompt(packet: dict, revision: str = PROMPT_VERSION, correction: str | None = None) -> str:
    prompt = WRITER_RULES + '\nAvailable information at the current time:\n' + json.dumps(packet, ensure_ascii=True, indent=2)
    if correction:
        prompt += '\nThe prior draft did not pass quality checks. Rewrite it in natural ENGLISH while addressing these specific issues. Do not invent an observation:\n' + correction
    return prompt


def semantic_prompt(packet: dict, annotation: dict) -> str:
    return SEMANTIC_RULES + '\n' + json.dumps({
        'kind': packet['kind'],
        'task': packet['task'],
        'text': annotation['text'],
        'probe_focus': packet.get('probe_focus', 'contact_target'),
        'next_action_intent': packet['next_demo_action_intent'],
        'next_control_mean_xyz': packet.get('planned_control_mean_xyz'),
    }, ensure_ascii=True, indent=2)
