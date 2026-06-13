import os
import json
import base64
import io
import sys
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Load model lazily
_model = None
_labels = None
_anchors = None

def get_model():
    global _model, _labels, _anchors
    if _model is None:
        print("Loading model...", flush=True)
        sys.path.insert(0, os.path.dirname(__file__))
        import model_setup
        import tensorflow as tf

        with open(model_setup.paths["labels.json"]) as f:
            _labels = json.load(f)
        with open(model_setup.paths["anchors.json"]) as f:
            _anchors = json.load(f)

        model_path = model_setup.paths["yolo_model.keras"]
        _model = tf.keras.models.load_model(model_path)
        print("Model loaded!", flush=True)
    return _model, _labels, _anchors


def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def run_detection(pil_img, obj_thresh=0.4, nms_thresh=0.45):
    from helpers import detect_image
    model, labels, anchors = get_model()
    result_img = detect_image(pil_img, model, anchors, labels, obj_thresh=obj_thresh, nms_thresh=nms_thresh)

    # Also collect detected objects with scores
    from helpers import preprocess_input, decode_netout, do_nms
    import numpy as np
    net_h, net_w = 416, 416
    image_w, image_h = pil_img.size
    new_image = preprocess_input(pil_img, net_h, net_w)
    yolo_outputs = model.predict(new_image)
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    boxes = do_nms(boxes, nms_thresh, obj_thresh)

    detections = []
    for box in boxes:
        detections.append({
            "label": labels[box.get_label()],
            "score": float(box.get_score()),
            "box": [box.xmin, box.ymin, box.xmax, box.ymax]
        })
    detections.sort(key=lambda x: x["score"], reverse=True)

    return result_img, detections


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        obj_thresh = float(data.get("obj_thresh", 0.4))
        nms_thresh = float(data.get("nms_thresh", 0.45))

        # image comes as base64
        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        result_img, detections = run_detection(pil_img, obj_thresh, nms_thresh)

        return jsonify({
            "result_image": "data:image/jpeg;base64," + pil_to_b64(result_img),
            "detections": detections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    try:
        import tempfile, cv2
        data = request.get_json()
        video_b64 = data["video"]
        if "," in video_b64:
            video_b64 = video_b64.split(",", 1)[1]
        video_bytes = base64.b64decode(video_b64)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
            tmp_in.write(video_bytes)
            tmp_in_path = tmp_in.name

        cap = cv2.VideoCapture(tmp_in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        import imageio
        writer = imageio.get_writer(tmp_out_path, fps=fps, codec="libx264", quality=5, macro_block_size=8)

        model, labels, anchors = get_model()
        from helpers import preprocess_input, decode_netout, do_nms, draw_boxes
        all_detections = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > 90:
                break
            if frame_count % 6 != 0:  # process every 3rd frame for speed
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                continue

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_w, image_h = pil_img.size
            net_h, net_w = 416, 416
            new_image = preprocess_input(pil_img, net_h, net_w)
            yolo_outputs = model.predict(new_image, verbose=0)
            boxes = decode_netout(yolo_outputs, 0.4, anchors, image_h, image_w, net_h, net_w)
            boxes = do_nms(boxes, 0.45, 0.4)

            result_pil = draw_boxes(pil_img, boxes, labels)
            result_frame = np.array(result_pil)
            writer.append_data(result_frame)

            for box in boxes:
                all_detections.append({
                    "label": labels[box.get_label()],
                    "score": float(box.get_score()),
                    "box": [box.xmin, box.ymin, box.xmax, box.ymax]
                })

        cap.release()
        writer.close()

        os.unlink(tmp_in_path)

        with open(tmp_out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp_out_path)

        # Aggregate detections
        agg = {}
        for d in all_detections:
            k = d["label"]
            if k not in agg or d["score"] > agg[k]["score"]:
                agg[k] = d
        final = sorted(agg.values(), key=lambda x: x["score"], reverse=True)

        return jsonify({
            "result_video": "data:video/mp4;base64," + video_b64,
            "detections": final
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/samples")
def samples():
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    files = []
    for f in os.listdir(sample_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            with open(os.path.join(sample_dir, f), "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            files.append({
                "name": f,
                "data": f"data:image/jpeg;base64,{b64}"
            })
    return jsonify(files)


@app.route("/api/status")
def status():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
