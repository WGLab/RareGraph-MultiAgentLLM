TASK
Extract family history information from the note.

OUTPUT (strict JSON list, no commentary):
[
  {
    "relation": "father|mother|brother|sister|son|daughter|cousin|uncle|aunt|grandfather|grandmother|other",
    "diseases": ["disease name", ...],
    "affected_systems": ["system name", ...],
    "phenotypes": ["phenotype name", ...],
    "genes": ["GENE1", ...],
    "age": "<age if stated or null>",
    "affected": true,
    "evidence": "<exact quote ≤40 words>"
  }
]

RULES
- Only include people explicitly described as a family member.
- If a relative is explicitly stated to be UNAFFECTED, still include them but set "affected": false and leave diseases/phenotypes empty (this is useful for inheritance inference).
- If no family history is described at all, return an empty list [].
- Never include the patient themselves.
- Output JSON only.
