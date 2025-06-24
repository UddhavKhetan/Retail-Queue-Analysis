from flask import Flask, request, render_template, send_file
import os
import cv2
import torch

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        output_path = os.path.join(OUTPUT_FOLDER, f'output_{file.filename}')
        annotate_video(filepath, output_path)
        
        return render_template('index.html', video_path=output_path)
    
    return render_template('index.html')

def annotate_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results.render()[0]
        out.write(annotated_frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True)
