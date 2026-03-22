# Results

## Building 1 (Verified)
`building1_results.json` — verified results from Google Colab run, March 2026.

## Buildings 2 and 3
Run the notebook (`notebooks/seismic_assessment_UTS_EGP42003.ipynb`) Cell 6
to generate results for all three buildings. JSON files will be saved in your
Colab session and can be downloaded from the Files panel.

## Result Summary (Building 1 verified)

| EDP | Value |
|-----|-------|
| T1 (FEM) | 0.610 s |
| T1 (code approx) | 0.288 s |
| PIDR Storey 1 | 0.230% |
| PIDR Storey 2 | 0.317% (governing) |
| AS1170.4 limit | 1.500% |
| Damage state | None |
| Compliance | PASS |
