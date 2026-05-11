# Data

This project uses a synthetic benchmark of outpatient-style clinical notes for follow-up instruction extraction.

The benchmark contains 2,000 synthetic notes generated from controlled structured skeletons. Each note includes:

- `note_text`
- `visit_date`
- scheduled follow-up actions
- temporal expressions
- canonical TestSpecification labels
- period dates / normalized day offsets
- character-span annotations for TestSpecification and TimeSpecification entities

The benchmark is synthetic and does not contain real patient data.

## Expected dataset format

The main dataset CSV is expected to include columns similar to:

```text
note_text
visit_date
actions_gt

where actions_gt is a JSON string containing entries such as:
[
  {
    "action": "CT Scan",
    "period_text": "in two weeks",
    "period_date": "2025-03-15",
    "action_char_start": 120,
    "action_char_end": 127,
    "time_char_start": 131,
    "time_char_end": 143
  }
]
