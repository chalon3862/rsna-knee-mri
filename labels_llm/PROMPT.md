You will read knee-MRI radiology reports and estimate, for each report, the probability that each of twelve
findings would be marked PRESENT by a musculoskeletal radiologist reviewing the IMAGES of that study.

Reports may be in English, Spanish, Turkish, Dutch, German, French, Bulgarian, Greek, Serbo-Croatian or other
languages. Read each report in its ORIGINAL language; do not translate it first.

## Output (strict)
For EVERY report, output exactly one line of JSON, in the same order as the reports, with:
- "id": the StudyInstanceUID given after ### ;
- for each of the 12 keys below: an INTEGER probability from 0 to 100 ;
- "mentioned": a list of the keys the report explicitly addresses (stated present OR stated absent/normal) ;
- "evidence": an object giving, for EVERY key you scored above 50, the short phrase of the report (copied in its
  original language, at most ~12 words) that supports it. If a value above 50 rests on an indirect sign (rule 5),
  quote that sign.
No other text, no explanations, no code fences.

Keys: ACL, MCL, MED_MENISCUS, LAT_MENISCUS, MED_OA, LAT_OA, PF_OA, EFFUSION, SYNOVITIS, BAKER, CONTUSION, FRACTURE

Example line (values are illustrative only):
{"id": "1.2.3", "ACL": 92, "MCL": 6, "MED_MENISCUS": 70, "LAT_MENISCUS": 12, "MED_OA": 18, "LAT_OA": 5, "PF_OA": 30, "EFFUSION": 65, "SYNOVITIS": 40, "BAKER": 4, "CONTUSION": 80, "FRACTURE": 10, "mentioned": ["ACL", "MED_MENISCUS", "EFFUSION", "CONTUSION"], "evidence": {"ACL": "rotura completa del LCA", "MED_MENISCUS": "rotura del cuerno posterior del menisco interno", "EFFUSION": "derrame articular moderado", "CONTUSION": "edema óseo en cóndilo femoral externo"}}

## How to score (read carefully)
1. Use the WHOLE 0-100 scale to express your confidence. Do NOT answer only 0/100 or a few fixed values.
   Two reports that differ in strength of evidence must get different numbers.
2. Ambiguous, equivocal, "cannot be excluded", "possible", "suspected", "compatible with", "probable" ->
   LOW-to-MIDDLE (roughly 15-45): the reference annotators mark ambiguous cases as NEGATIVE.
3. Finding EXPLICITLY stated absent / normal / intact -> very low (1-5).
4. Finding NOT MENTIONED at all -> do NOT use a fixed default. Estimate it from the rest of the report and from
   how findings co-occur, e.g.:
   - SYNOVITIS is rarely written down but usually accompanies a moderate/large effusion, Hoffa / suprapatellar
     inflammation, or inflammatory arthropathy -> raise it with effusion size and inflammatory wording.
   - OA (MED/LAT/PF) not named but osteophytes, joint-space narrowing, chondral thinning/defects, subchondral
     sclerosis/cysts in that compartment -> raise that compartment accordingly.
   - CONTUSION / FRACTURE not named but acute trauma pattern (ACL tear, bone marrow edema, "bone bruise",
     pivot-shift pattern) -> raise contusion; fracture only with fracture wording or rule 5.
   - A short, normal-sounding report with nothing abnormal -> all unmentioned findings low (2-10).
   Different unmentioned contexts MUST give different numbers.
5. INDIRECT SIGNS (classic associations; apply them, and quote the sign in "evidence"):
   - Lipohemarthrosis / fat-fluid level -> intra-articular fracture likely: FRACTURE high (75-90) even if no
     fracture line is described.
   - Segond fracture (lateral tibial rim avulsion) -> ACL tear very likely: ACL high (80-95) if the ACL is not
     described as intact.
   - Pivot-shift bone-bruise pattern (lateral femoral condyle + posterolateral tibial plateau) with the ACL NOT
     described -> ACL moderately raised (40-60). If the ACL is described as intact, rule 3 wins.
   - Subchondral insufficiency fracture (SIFK / SONK with a fracture line) -> FRACTURE (counts as a stress
     fracture).

## Reading rules
6. MEDIAL = INTERNAL, LATERAL = EXTERNAL, in every language: e.g. interne / externe (fr), interno / externo (es,
   it, pt), iç / dış (tr), binnen / buiten (nl), innen / außen (de). "Fémoro-tibial interne", "femorotibial
   medial", "compartimento interno" = medial tibiofemoral compartment.
7. Synonyms count in every language: popliteal cyst / kyste poplité / quiste de Baker / Baker kisti /
   semimembranosus-gastrocnemius bursa distension = BAKER ; bone marrow edema / œdème osseux / edema óseo /
   kemik iliği ödemi / beenmergoedeem / Knochenmarködem / bone bruise / contusion = CONTUSION (traumatic pattern);
   chondropathy / chondromalacia / condropatía / kondromalazi = cartilage damage (see OA definitions).
8. Negation and doubt in every language: "sans", "absence de", "no se observa", "sin", "izlenmedi", "yok",
   "saptanmadı", "geen", "kein", "nicht" -> absent ; "non exclu", "no se descarta", "dışlanamaz", "niet uit te
   sluiten", "nicht auszuschließen", "verdacht", "suspect" -> ambiguous (rule 2).
9. If the findings section and the conclusion / impression disagree, follow the MORE SPECIFIC statement (the one
   that names location, grade or pattern).
10. Meniscal tear WITHOUT a stated side (e.g. "posterior horn tear", "meniscal tear") -> split the uncertainty
    between MED_MENISCUS and LAT_MENISCUS (both middle values), unless anatomy in the sentence fixes the side.
11. Post-operative knee (meniscectomy, ACL reconstruction, osteotomy, arthroplasty): score the CURRENT state;
    a surgically treated structure without a NEW lesion is low-middle (meniscus) or low (intact graft).

## Grading scales -> probability
12. Meniscus (Stoller / Mesgarzadeh): grade 1 or 2 intrasubstance signal (not reaching an articular surface)
    -> low (10-25) ; grade 3 / signal reaching the surface / "tear" -> high (75-95).
13. Ligaments (ACL / MCL): grade I / sprain / edema around an intact ligament -> low (10-25) ; grade II / partial
    tear / high-grade partial -> middle-high (55-80) ; grade III / complete tear / discontinuity -> very high
    (90-98). ACL "mucoid degeneration" / "ganglion" without tear -> low (10-20).
14. Cartilage and OA, per compartment: Outerbridge / ICRS grade 1 -> low (10-20) ; grade 2 -> low-middle (30-45) ;
    grade 3 -> high (65-80) ; grade 4 / full-thickness loss / bone exposure -> very high (85-95). Kellgren-
    Lawrence 1 -> low ; 2 -> middle ; 3-4 -> high. Osteophytes + narrowing in a compartment -> high for it.

## Definitions (what counts as PRESENT)
- ACL / MCL: a tear (complete OR partial / high-grade) — see rule 13. Reconstructed ACL graft that is intact ->
  low; graft tear -> high.
- MED_MENISCUS / LAT_MENISCUS: a TEAR (any pattern, incl. root tear, flap, bucket-handle, complex, radial,
  horizontal). "Degeneration", "intrasubstance signal" WITHOUT tear -> low (rule 12). Post-meniscectomy without
  new tear -> low-middle.
- MED_OA / LAT_OA / PF_OA: osteoarthritis / degenerative joint disease of THAT compartment (tibiofemoral medial,
  tibiofemoral lateral, patellofemoral) — see rule 14. A generic "degenerative changes" gives a moderate value to
  the compartments it most likely concerns.
- EFFUSION: joint effusion. "Physiological" / "trace" / "minimal" fluid -> low (10-25); "small" effusion ->
  middle-high (50-70); moderate / large -> high (80-98).
- SYNOVITIS: synovial thickening / synovitis / hypertrophic synovium (see rule 4 when not mentioned).
- BAKER: popliteal (Baker's) cyst, any size; ruptured cyst counts. NOT a Baker's cyst: parameniscal cyst,
  ganglion cyst, bursitis other than the semimembranosus-gastrocnemius bursa.
- CONTUSION: bone marrow edema / bone bruise / contusion of traumatic pattern. Degenerative subchondral edema,
  edema with osteoarthritis, or reactive edema around a meniscal root tear only -> low-middle.
- FRACTURE: any fracture incl. avulsion (e.g. Segond), osteochondral, impaction, stress or subchondral
  insufficiency fracture (rule 5). Bone bruise alone is NOT a fracture.

Reports follow, each after a line "### <StudyInstanceUID>".
