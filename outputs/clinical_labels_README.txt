BioVision Clinical Annotation Instructions

1. filename must exactly match the filename in outputs/fused_features.csv.
2. healing_outcome must come from genuine clinical annotation/data.
3. infection_status must come from genuine clinical annotation/data.
4. Labels must not be inferred from image appearance or fabricated.
5. Save the final populated file as outputs/clinical_labels.csv.
6. After the real file is provided, run label_validation.py before any model training.

The template contains the five existing integration-test filenames. The clinical label fields are intentionally empty.
