TASK
Identify observable facial / dysmorphic phenotypes from the patient photograph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLOSED VOCABULARY (pick ONLY from this list)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Eyes / periorbital:
  hypertelorism, hypotelorism, telecanthus, upslanting palpebral fissures,
  downslanting palpebral fissures, long palpebral fissures, short palpebral fissures,
  epicanthus, ptosis, proptosis, deep-set eyes, almond-shaped eyes

Ears:
  low-set ears, posteriorly rotated ears, prominent ears, preauricular tag,
  microtia, cup-shaped ears

Nose / philtrum:
  broad nasal bridge, flat nasal bridge, depressed nasal bridge, prominent nose,
  bulbous nasal tip, anteverted nares, short philtrum, long philtrum, smooth philtrum

Mouth / jaw:
  thin upper lip, full lower lip, cupid's bow, tented upper lip,
  micrognathia, retrognathia, macrognathia, cleft lip, cleft palate

Forehead / head shape:
  high forehead, prominent forehead, sloping forehead, frontal bossing,
  microcephaly, macrocephaly, brachycephaly, dolichocephaly, plagiocephaly,
  trigonocephaly, turricephaly

Face shape / features:
  coarse facies, flat facies, triangular face, long face, round face, narrow face,
  midface retrusion, malar hypoplasia

Eyebrows / hair:
  synophrys, thick eyebrows, sparse eyebrows, sparse hair, widow's peak,
  low posterior hairline, low anterior hairline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT (strict JSON list, no commentary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  {
    "mention": "<phenotype name from the closed vocabulary above>",
    "confidence": "high|medium|low",
    "evidence": "<short description of the visual cue>"
  }
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use ONLY phenotypes from the list above. Do NOT invent new terms.
- Do NOT attempt to name a syndrome or disease.
- If no clearly observable dysmorphic features, return [].
- Do NOT output confident findings for subtle or uncertain features; use "low".
- Output JSON only.
