import os
import torch
from flask import Flask, request, send_from_directory, redirect, url_for
from flask import render_template
from werkzeug.utils import secure_filename
from utils.recognizer import ActionRecognizer, process_video_with_actions

# Adjust paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_PATH = 'models/action_model_full.pth'
HTML_PATH = 'static/home.html'

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your model
recognizer = ActionRecognizer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recognizer.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
recognizer.model.eval()

@app.route('/')
def home():
    # Serve static HTML directly
    return send_from_directory('static', 'home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Secure filename and save the file
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f"action_{filename}"
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

    file.save(input_path)

    # Process the video
    try:
        process_video_with_actions(input_path, output_path, recognizer)
    except Exception as e:
        return f"Error processing video: {str(e)}", 500

    # Redirect to the result page
    return redirect(url_for('result', filename=output_filename))

@app.route('/results/<filename>')
def result(filename):
    # Serve the processed video result
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
