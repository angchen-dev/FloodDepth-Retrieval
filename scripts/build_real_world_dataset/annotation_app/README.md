# Real-world flood-depth annotation app

This FastAPI/Three.js web app was used in the study *Flood depth estimation using images of vehicles: a new image retrieval framework with annotated datasets* to collect annotated flood depths for the real-world dataset. Participants select a similar 3D vehicle and align a virtual water plane with the waterline in each flood image. This public version is a lightweight demonstration of that annotation workflow, and its entries are stored in a local SQLite database.

## Data layout

Download the repository data bundle before starting the app. The default paths are:

```text
data/build_real_world_dataset/
├── images/                                 # annotation and trial images
└── assets/
    ├── car_images/                         # vehicle thumbnails
    ├── car_models/                         # matching GLB models
    ├── instruction/                        # instruction images
    └── textures/                           # HDR environment texture
```

The included `annotation_groups.json` is a lightweight demo configuration with two annotation groups of five images each: `051.jpg`–`055.jpg` and `056.jpg`–`060.jpg`. Images `trial_049.jpg`, `trial_050.jpg`, and `trial_201.jpg` are used as trials. These 13 images are sufficient to run the demo; other files in the image directory are ignored unless they are added to the configuration.

This demo preserves the annotation workflow but does not reproduce the full participant image allocation used for the paper study. Edit `annotation_groups.json` to define a different set of annotation groups.

The opening page is a demo notice rather than a research consent form. It intentionally contains no personal researcher name or email address. Before using the application to collect data from participants, replace the notice with an institutionally approved participant information and consent form and review the complete data-handling workflow.

## Install and run

Run from the repository root:

```bash
conda activate fd_retrieval_env
python -m pip install -r scripts/build_real_world_dataset/annotation_app/requirements.txt
python scripts/build_real_world_dataset/annotation_app/main.py
```

Open <http://127.0.0.1:8080>. Use `--host 0.0.0.0` only when the app must be reachable from another machine on a trusted network.

The database is created at `scripts/build_real_world_dataset/annotation_app/results/results.db`. Runtime results are not intended to be committed.

## Process the results

After annotations have been collected, start Jupyter from the repository and open the processing notebook:

```bash
jupyter lab scripts/build_real_world_dataset/process_results.ipynb
```

Run all cells to create `filtered_annotations.csv`, `real_world_gt_info.json`, and `real-world_level_frequency.pdf` under `scripts/build_real_world_dataset/annotation_app/results/processed/`. The notebook filters formal annotations using the paper's three-annotation outlier rule and plots the final image-level depths in 10 cm intervals.

## Optional path overrides

The following environment variables override the defaults:

- `FLOOD_APP_IMAGES_DIR`: directory containing real-world images.
- `FLOOD_APP_ASSETS_DIR`: directory containing the four asset subdirectories.
- `FLOOD_APP_DB_PATH`: SQLite output path.

The app loads Bootstrap and Three.js from jsDelivr, so the browser requires internet access unless those frontend dependencies are vendored locally.
