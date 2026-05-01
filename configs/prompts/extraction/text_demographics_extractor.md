TASK
Extract patient demographics from the clinical note below.

OUTPUT FORMAT (strict JSON, no other text):
{
  "age": {"value": "<age as written, or null>", "age_group": "<prenatal|neonatal|infantile|childhood|adolescent|adult|elderly|unknown>"},
  "sex": {"value": "<male|female|unknown>"},
  "ethnicity": {"value": "<ethnicity/race as written, or null>"}
}

RULES
- Only extract demographics belonging to THE PATIENT (the proband).
- Do NOT extract demographics of family members.
- If a field is not mentioned, set its value to null and age_group/sex to "unknown".
- Output JSON only. No commentary.
