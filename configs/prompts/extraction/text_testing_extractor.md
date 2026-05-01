TASK
Extract diagnostic tests/investigations performed on THE PATIENT (proband).

OUTPUT (strict JSON list, no commentary):
[
  {
    "test_name": "<name>",
    "category": "laboratory|imaging|electrophysiology|genetic|pathology|microbiology|functional|clinical_assessment|other",
    "value": "<numeric or textual value or null>",
    "unit": "<unit or null>",
    "result_interpretation": "normal|abnormal|elevated|decreased|positive|negative|pending|unknown",
    "date": "<date or null>",
    "evidence": "<exact supporting sentence>"
  }
]

RULES
- Only tests belonging to the patient.
- If none, return [].
- Output JSON only.
