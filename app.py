from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import os
import cv2
import logging
from ssd_model import SSDModel
from video_stream import generate_frames

# Initialize the model (ensure the path is correct)
model = SSDModel('/Users/wynjehu/Desktop/Sheran/ssd_mobilenet_v2_320x320_coco17_tpu-8 3/saved_model')

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)

# Example label map (replace with your actual label map)
LABEL_MAP = {
    1: 'Person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'aeroplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'pottedplant',
    65: 'bed',
    67: 'diningtable',
    70: 'toilet',
    72: 'tvmonitor',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
    # Add more class mappings as needed
}

def detect_objects(frame):
    # Convert BGR to RGB for TensorFlow model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, scores, classes = model.detect_objects(rgb_frame)

    # Debugging: Print the shapes of the detection results
    logging.debug(f"Boxes shape: {boxes.shape}")
    logging.debug(f"Scores shape: {scores.shape}")
    logging.debug(f"Classes shape: {classes.shape}")

    # Process each detection
    height, width, _ = frame.shape
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            # Get class name from label map
            class_id = int(classes[i])  # Ensure class_id is an integer
            class_name = LABEL_MAP.get(class_id, 'Unknown')

            # Draw class name and confidence score on the frame
            label = f"{class_name}: {scores[i]:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return boxes, scores, classes

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

import subprocess

def reencode_video(input_path, output_path):
    # Use FFmpeg to re-encode the video to ensure compatibility
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"FFmpeg output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        raise

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
        return jsonify({'error': 'Unsupported file type. Please upload a video file.'}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    output_path = os.path.join(OUTPUT_FOLDER, f'processed_{os.path.splitext(video_file.filename)[0]}.mp4')

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read video frame")

        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Alternative codec
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame (e.g., object detection)
            boxes, scores, classes = detect_objects(frame)

            out.write(frame)

        cap.release()
        out.release()

        # Re-encode the video to ensure compatibility
        reencoded_output_path = os.path.join(OUTPUT_FOLDER, f'reencoded_{os.path.splitext(video_file.filename)[0]}.mp4')
        reencode_video(output_path, reencoded_output_path)

        return jsonify({'download_url': f'/outputs/{os.path.basename(reencoded_output_path)}'})

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)