# Retail Shelf Price Tag Detection & OCR Pipeline

A complete end-to-end computer vision system for detecting supermarket shelf price tags, cropping them, reading the printed price using OCR, and producing structured outputs in JSON format. The pipeline is engineered to operate under realistic retail conditions such as blur, lighting glare, small and distant tags, mixed English–Japanese text, and inconsistent tag layouts.

# -> Project Directory Structure
root
├── Sample Dataset/
├── Details.docx
├── evals/
├── plan/
│
├── project/
│   ├── config/
│   ├── data/
│   │   ├── raw/
│   │   ├── crops/
│   │   ├── results/
│   │   └── annotations/
│   ├── detection/
│   ├── eval/
│   ├── final_outputs/
│   ├── logs/
│   ├── models/
│   ├── ocr/
│   ├── pipeline/
│   ├── training/
│   ├── utils/
│   │
│   └── run_pipeline_on_shelf_image.py
│
├── yolov8n.pt
├── setup_proj.py
└── README.md


All core implementation (training, inference, OCR, evaluation) resides inside the project/ directory.

# -> Methodology — System Overview

The input shelf image is first auto-oriented and resized to match the preprocessing pipeline used during training. A multi-scale YOLOv8 detector is then applied across several image scales to improve recall on small and distant price tags. Detections from all scales are merged using cross-scale Non-Max Suppression. Each detected region is expanded slightly and cropped at high resolution.

These crops are passed through a dual-language OCR stage (English + Japanese). The OCR output is re-ranked using confidence and relative text height, exploiting the observation that price values tend to be the largest printed text on a tag. A price-specific parsing module extracts numerical values while separating product name and auxiliary fields. The final structured JSON record is written per tag.

Image → Preprocess → Multi-Scale YOLO → Merge Detections
 → High-Res Crop → Dual-Language OCR → Price Parsing → JSON Output


The design focus is on robustness and practical detection performance rather than heavy retraining or synthetic augmentation-driven optimization.

# -> Model & Training Overview

A pretrained yolov8n.pt model was fine-tuned on a small, manually annotated dataset exported via Roboflow. Training emphasized accurate bounding box supervision over aggressive augmentation so that localization remained stable. Key training characteristics included 640×640 input resolution, 60 epochs, batch size 8, and a single-class head (price_tag).

Rather than relying only on model training to improve results, most performance gains were obtained through inference-time engineering: multi-scale detection, relaxed confidence thresholds, bounding-box expansion before OCR, and stronger OCR post-processing.

This hybrid strategy is suitable for small-data retail environments where annotation resources are limited but inference control is flexible.

# -> Evaluation Summary

Evaluation was performed on five manually labeled shelf images comprising 93 ground-truth price tags. Results reflect realistic noisy conditions.

Metric	     Score
Detection    Recall	0.86
OCR Price    Accuracy	0.91
End-to-End   Accuracy	0.79

The pipeline demonstrates strong OCR reliability and competitive detection recall, especially for mid-size and clear tags.

# ->Tech Stack

The system uses Ultralytics YOLOv8 for detection, OpenCV and NumPy for preprocessing and cropping, and EasyOCR for multilingual text recognition. Ground-truth evaluation and structured output processing are implemented using custom Python scripts with JSON interfaces.

# -> Known Failure Cases & Observations

Despite robustness improvements, several realistic challenges remain:

Very tiny tags at extreme distances
When price tags occupy only a few pixels at source resolution, even multi-scale detection struggles. In such cases, the bounding box either collapses or merges with neighboring text. Tiling-based inference could further improve recall, but increases runtime.

Heavy motion blur or strong glare
Crops become unreadable for OCR even when detection is correct. Since OCR operates on the crop rather than the resized training resolution, image quality directly affects price extraction.

Stacked or overlapping price labels
Adjacent labels may merge into one detection box at certain scales. Cross-scale NMS reduces this, but very dense shelves still cause occasional merging.

Non-price numeric elements misinterpreted as price
Although height-based ranking mitigates this, some layouts place numeric codes or promotional text in similar font sizes. The structured price parser handles most such cases but is not perfect.But this is rare.

Overall, failures are dominated by image quality and scale effects, rather than OCR parsing logic or model misclassification.

# -> Running the Final Pipeline

To run inference on a shelf image:

python project/run_pipeline_on_shelf_image.py


The script saves:

detection visualizations

cropped price tag images

structured JSON output

All outputs are written to project/final_outputs/.
