# eval_style fixtures

These fixtures are small, hand-written fake Chinese text samples for deterministic regression tests.

They do not contain real novel source text, real model output, or private project material. They are safe to keep in git and are only used by `_test_eval_style.py`.

- `reference_basic.txt`: baseline fake novel-style reference.
- `candidate_close.txt`: similar form and rhythm, different content.
- `candidate_repetition.txt`: obvious repeated sentences and short sentence loops.
- `candidate_contamination.txt`: deliberately copies text from the reference.
- `candidate_far.txt`: intentionally different technical/instructional style.
- `candidate_empty.txt`: blank candidate used to verify invalid scoring.
