TASK
Extract every phenotype or clinical finding from the note below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sentences in the note may be preceded by inline flags computed by a preprocessor:
- [NEG]         → sentence contains negation triggers
- [FAMILY]      → sentence references a family member
- [HYPOTHETICAL] → sentence contains uncertainty (if, possible, concern for, may have, suspected, rule out)
- [HISTORICAL]  → sentence mentions historical context

Use these flags as strong hints for the attribution field, but still read the full sentence to make the final decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. mention
   The phenotype term only, stripped of negation/uncertainty wrappers.

   INCLUDE:
   - Symptoms, dysmorphic features, developmental findings, lab/imaging/EEG findings,
     functional limitations, cognitive/behavioral findings.

   EXCLUDE:
   - Disease names used purely as diagnoses (not phenotypes)
   - Medications, treatments, procedures

   NORMALIZATION FOR NEGATED / NORMAL FINDINGS:
   - "normal X" → convert mention to the abnormal counterpart
     e.g., "normal muscle tone" → mention = "abnormal muscle tone"
     e.g., "normal hearing" → mention = "hearing loss"

2. attribution (exactly one of):
   "patient" | "negated" | "family" | "references" | "others" | "uncertain"

   RULES (apply in order; first matching wins):
   Rule 1 — NEGATED: denied / not present / normal / absent / resolved
   Rule 2 — UNCERTAIN: possible / suspected / concern for / rule out / ?
   Rule 3 — FAMILY: applies to a relative (mother, father, sister, brother, aunt, etc.)
   Rule 4 — REFERENCES: general disease description / educational content
   Rule 5 — OTHERS: applies to a non-family third party
   Rule 6 — PATIENT: none of the above; clearly affirmed for the patient

3. onset
   Short text only ("at birth", "infantile", "6 months", "since age 2"). null if not stated.

4. evidence
   Verbatim quote from the note, ≤10 words, supporting mention + attribution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPLITTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"no history of X, Y, or Z"     → 3 negated entries
"with A, B, and C"              → 3 patient entries
"mother has X and Y"            → 2 family entries
"possible X or Y"               → 2 uncertain entries

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return a JSON list only (no commentary, no markdown fences):
[
  {"mention": "...", "attribution": "patient|negated|family|references|others|uncertain",
   "onset": "<short|null>", "evidence": "<≤10 word quote>"}
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTE TO PROCESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
