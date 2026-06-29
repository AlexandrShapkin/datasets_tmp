# Temporary dataset builder

This repository contains a reproducible GitHub Actions pipeline for building a three-class image-classification dataset:

- `feijoa`
- `carrot`
- `watermelon`

The builder downloads independently sourced, openly licensed photographs, validates them, removes exact and perceptual duplicates, ranks content with CLIP, selects a diverse subset, and produces a ZIP archive with `train`, `val`, and `test` directories.

No rotated, mirrored, recolored, or otherwise augmented copies are included as separate dataset samples.

The generated artifact contains 100 unique source photographs per class, a source and license manifest, validation reports, and contact sheets for visual inspection.
