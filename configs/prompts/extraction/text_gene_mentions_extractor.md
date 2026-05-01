TASK
Extract gene/variant findings belonging to THE PATIENT (proband) only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE ONE RULE THAT CANNOT BE BROKEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT extract genes belonging to a family member.
DO NOT extract genes belonging to a family member.
DO NOT extract genes belonging to a family member.

Family terms include: mother, father, parent, brother, sister, sibling, aunt, uncle,
cousin, grandmother, grandfather, son, daughter, relative, maternal, paternal,
"in the family", "family history of", and any other phrasing that refers to someone
other than the patient.

If the sentence contains any family term AND a gene, the gene goes to family history
(not here). Use context flags [FAMILY] to help you.

If you are UNSURE who the gene belongs to → do NOT extract it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT (strict JSON list, no commentary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  {
    "gene": "<exact gene symbol from the note>",
    "variant": "<variant string or null>",
    "zygosity": "homozygous|heterozygous|compound_heterozygous|hemizygous|mosaic|unknown",
    "result": "positive|negative|inconclusive|pending|unknown",
    "evidence": "<exact sentence proving this gene is the PATIENT's>"
  }
]

If no patient-owned gene is mentioned, return [].
Output JSON only.
