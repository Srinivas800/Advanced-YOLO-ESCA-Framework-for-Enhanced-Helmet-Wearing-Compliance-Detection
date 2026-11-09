# YOLO-ESCA: Safety Helmet Detection

A real-time helmet detection model built on YOLOv5, designed for UAV and video stream deployment on construction and industrial sites.

## What it does

Detects whether workers are wearing safety helmets. The model finds three classes in each frame:

- `head`: head without a helmet
- `helmet`: head with a helmet
- `person`: full person

## Results

| Metric | Value |
|---|---|
| mAP@0.5 | 94.7% |
| Speed | 65.3 FPS |
| Training images | 4,400 (self-built dataset) |
| Classes | 3 |

Compared with the original YOLOv5n:

- Detected **41.7% more objects** on the self-built dataset
- Detected **73% more objects** on the public SHWD benchmark (Safety Helmet Wearing Dataset)

## Improvements over YOLOv5

1. **CBAM attention**: helps the model focus on small and partly hidden helmets.
2. **EIoU loss**: gives tighter bounding boxes by penalizing width and height errors separately.
3. **Soft-NMS**: lowers the confidence of overlapping boxes instead of deleting them, which helps when workers stand close together.
4. **Small-target detection layer**: an extra detection head for tiny objects seen from UAV height.

## Dataset

- Self-built dataset of 4,400 images with 3 classes.
- This repo includes a sample: 761 training images and 218 validation images.
- Labels are in YOLO format: `class x_center y_center width height`, normalized from 0 to 1.
- Also evaluated on SHWD, which was used only for testing and not for training.

`Data/dataset.yaml`:

```yaml
nc: 3
names: ['head', 'helmet', 'person']
```

Update the `train` and `val` paths in this file to match your own folder locations.

## Project structure

```
YOLO-ESCA/
├── Data/
│   ├── dataset.yaml
│   └── images/
│       ├── train/
│       └── valid/
└── notebook.ipynb
```

## How to run

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

python train.py \
  --img 640 \
  --batch 16 \
  --epochs 80 \
  --data ../Data/dataset.yaml \
  --cfg path/to/your_yolo_esca.yaml \
  --weights ''
```

Replace `path/to/your_yolo_esca.yaml` with your YOLO-ESCA model config.

## Tech stack

Python, PyTorch, YOLOv5 (Ultralytics), Google Colab (GPU)

## Reference

*YOLO-ESCA: High-Performance Safety Helmet Standard Wearing Behaviour Detection Model Based on Improved YOLOv5*
