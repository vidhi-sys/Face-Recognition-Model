import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pickle
import sys

# Add the directory containing face_recognition.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_recognition import face_recognition # Import your function

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # List of example images already in static/uploads
    # You might want to dynamically list them or just hardcode if fixed
    example_images = [
        'eigen_0.jpg', 'eigen_1.jpg', 'eigen_2.jpg', 'eigen_5.jpg',
        'eigen_6.jpg', 'eigen_7.jpg', 'roi_1.jpg', 'roi_2.jpg',
        'roi_3.jpg', 'roi_6.jpg'
    ]
    return render_template('index.html', uploaded_image=None, predictions=None, example_images=example_images)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error_message="No file part in the request.", example_images=get_example_images())

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error_message="No selected file.", example_images=get_example_images())

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return process_and_render(filepath, filename)
    else:
        return render_template('index.html', error_message="Invalid file type. Allowed: png, jpg, jpeg, gif", example_images=get_example_images())

@app.route('/process_example/<filename>')
def process_example_image(filename):
    # Ensure the filename is secure and is one of your pre-approved examples
    secure_name = secure_filename(filename)
    if secure_name not in get_example_images(): # Basic check against hardcoded list
        return render_template('index.html', error_message="Invalid example image.", example_images=get_example_images())

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
    if not os.path.exists(filepath):
        return render_template('index.html', error_message="Example image not found.", example_images=get_example_images())

    return process_and_render(filepath, secure_name)

def process_and_render(filepath, original_filename):
    """Helper function to process image and render template."""
    try:
        processed_img, predictions_data = face_recognition(filepath)

        # Save the processed image to display it
        # Append a unique identifier to avoid overwriting processed images with same name
        processed_filename = f"processed_{os.path.basename(filepath)}"
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, processed_img) # Save the processed OpenCV image

        return render_template('index.html',
                               uploaded_image=url_for('static', filename=f'uploads/{processed_filename}'),
                               predictions=predictions_data,
                               original_filename=original_filename, # Pass original filename for display
                               example_images=get_example_images())
    except Exception as e:
        return render_template('index.html', error_message=f"Error processing image: {e}", example_images=get_example_images())

def get_example_images():
    """Helper to get list of example images. Could be dynamic in a real app."""
    # This should match the files you manually placed in static/uploads
    return [
        'eigen_0.jpg', 'eigen_1.jpg', 'eigen_2.jpg', 'eigen_5.jpg',
        'eigen_6.jpg', 'eigen_7.jpg', 'roi_1.jpg', 'roi_2.jpg',
        'roi_3.jpg', 'roi_6.jpg'
    ]


if __name__ == '__main__':
    app.run(debug=True)