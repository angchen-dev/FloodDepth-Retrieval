# Build the real-world dataset

The annotation tool is in `annotation_app/`. The app was used in the study *Flood depth estimation using images of vehicles: a new image retrieval framework with annotated datasets* to collect annotated flood depths for the real-world dataset. The bundled configuration contains two groups of five annotation images for a lightweight demo rather than the full paper collection. See `annotation_app/README.md` for setup, data layout, and usage.

After collecting annotations, open `process_results.ipynb` to filter the annotations, create the ground-truth JSON, and reproduce the real-world flood-depth distribution figure. The notebook writes a filtered CSV, a JSON file, and a PDF figure to `annotation_app/results/processed/`.
