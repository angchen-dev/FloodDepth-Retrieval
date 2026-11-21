# DSC-YOLOv8n baseline results

This directory does not redistribute the DSC-YOLOv8n implementation, training dataset, or the checkpoint trained for our reproduction. The paper states that its code and data are available from the original authors upon request. Researchers who need these materials should contact the authors and follow their access and usage conditions.

Original paper:

- [DSC-YOLOv8n: An advanced automatic detection algorithm for urban flood levels](https://www.sciencedirect.com/science/article/pii/S0022169424014240), *Journal of Hydrology*, 643, 132028 (2024). DOI: `10.1016/j.jhydrol.2024.132028`.

## Paper prediction files

The predictions produced by our reproduction and used in the paper comparison are stored with the other runtime results:

- `runs/paper/baselines/dsc_yolov8n/real_world/results.json`
- `runs/paper/baselines/dsc_yolov8n/synthetic/results.json`

These are experimental outputs from our reproduction, not results or software released by the DSC-YOLOv8n authors. The result-analysis notebook reads these files directly. For the paper protocol, when multiple vehicles were detected, the candidate with the smallest absolute error to the ground truth supplied the image-level prediction. All candidates are retained in each JSON record.
