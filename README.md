# StreetScan

A real-time road object detection web app powered by YOLOv3 and TensorFlow, built as the culmination of a 2-week AI program through Inspirit AI.

---

## About

Working alongside my Inspirit AI cohort, I researched and evaluated multiple computer vision approaches — including CNN sliding window models, VGG16, VGG19, and DenseNet121 — before landing on YOLO for its speed and accuracy. I then built and deployed the full web application on top of the trained model.

StreetScan detects vehicles, pedestrians, and road obstacles from images and webcam video, while simulating how an autonomous vehicle would respond to the scene.

---

## Features

- **Image Detection** — Upload any road image and detect objects with bounding boxes and confidence scores
- **Video Detection** — Record a webcam clip and run object detection frame by frame
- **Autonomous Driving Analysis** — Simulates real-time decision-making (Continue, Slow Down, Stop Immediately)
- **Threshold Explorer** — Adjust the objectness threshold interactively to see how detection sensitivity changes
- **Sample Images** — Built-in road scene examples to test the model instantly

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, TensorFlow, OpenCV |
| Model | YOLOv3 |
| Frontend | HTML, CSS, JavaScript |
| Video Processing | imageio, imageio-ffmpeg |
| Deployment | Railway |

---

## Running Locally

1. Clone the repo:
```bash
git clone https://github.com/alex-hou-09/RoadDetector.git
cd RoadDetector
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```

4. Open your browser at `http://localhost:5000`

---

## Project Context

This project was built during a 2-week intensive AI program through **Inspirit AI** (June 2026). The model research and training was a collaborative effort with my cohort, while the web application and deployment were built independently.

**Cohort members:** Akhil Desai, Jayash Patnaik, Alex Hou

---

## License

MIT
