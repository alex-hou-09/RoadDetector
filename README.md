# RoadDetector

Real-time road object detection powered by YOLOv3.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open http://localhost:5000

Note: On first run, the model (~250MB) will download automatically from HuggingFace.
