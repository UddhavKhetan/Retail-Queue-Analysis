from flask import Flask, render_template, request, send_from_directory
import os
from yolov8_script import process_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['video']
        if uploaded_file.filename != '':
            input_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(input_path)

            output_video = os.path.join(OUTPUT_FOLDER, 'output.mp4')
            output_csv = os.path.join(OUTPUT_FOLDER, 'queue_time.csv')

            process_video(input_path, output_video, output_csv)

            return render_template('index.html',
                                   video_file='static/output.mp4',
                                   csv_file='static/queue_time.csv')
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
