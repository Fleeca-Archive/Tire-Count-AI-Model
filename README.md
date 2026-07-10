# Tire Count AI Model

## Project Overview

This project builds an intelligent computer vision model capable of detecting and automatically counting tires in images. The system analyzes visual data to identify individual tire occurrences within photographs, providing accurate quantification for inventory management, warehouse operations, and logistics analysis.

## Objective

Develop a robust object detection and counting system that:
- Automatically identifies tires within images regardless of scene complexity
- Accurately counts the number of tires present in a single photograph
- Provides precise localization of each detected tire through bounding box or polygon annotations
- Handles diverse environmental conditions and tire orientations
- Scales reliably across various warehouse and storage environments

## Dataset

- **Initial collection**: Raw tire images from various operational environments
- **Cleaned dataset**: Curated high-quality photos with removed duplicates and corrupted images
- **Challenge addressed**: Handling variable lighting conditions, overlapping tires, partial visibility, shadows, and background clutter
- **Annotation format**: Polygon annotations using Label Studio marking individual tire locations

## Project Structure

```
├── labels.csv                   # Master annotation registry with tire counts and coordinates
├── Count Tyres/                 # Test and validation images
│   ├── 01.jpg to 06.jpg        # Indexed test images
│   └── [UUID-named images]     # Annotated dataset images
├── images.zip                   # Compressed source images
├── tire_count_model.h5         # Trained object detection model
├── README.md                    # This file
└── .git/                        # Version control history
```

## Pipeline Steps

### Step 1: Image Collection & Dataset Curation
Gathered diverse tire images across multiple warehouse and storage scenarios. Applied rigorous quality control to eliminate:
- Motion blur and camera shake artifacts
- Extreme lighting variations and shadows
- Corrupted or unreadable image files
- Severely occluded or partially visible tires
- Non-tire objects and background noise

### Step 2: Precise Annotation & Labeling
- **Tooling**: Label Studio annotation platform
- **Strategy**: Tight polygon outlines drawn around each individual tire perimeter
- **Methodology**: Mark every distinct tire occurrence with high-precision coordinates
- **Output**: Ground-truth registry linking image filenames to tire counts and boundary coordinates

**Annotation Standards**:
- Use precise polygon vertices for each tire boundary (not broad bounding boxes)
- Trace complete tire outline including rim and tread area
- Count and annotate every visible tire including partially obscured instances
- Record exact pixel coordinates for each tire detection
- Maintain consistent annotation quality across all annotators
- Document any ambiguous tire instances or occlusions

### Step 3: Data Organization & CSV Registry
Compiled all annotation metadata into `labels.csv` containing:
- Unique annotation IDs and image references
- Polygon coordinate arrays for precise localization
- Tire count metadata extracted from polygon annotations
- Image dimensions and orientation information
- Annotation timestamps and curator information

### Step 4: Image Preprocessing Pipeline
- Automated pipeline extracts image files from source archives
- References polygon coordinates from master CSV
- Crops or segments individual tire regions based on polygon boundaries
- Normalizes image dimensions for model training consistency
- Generates training/validation splits

### Step 5: Object Detection Model Training
- **Architecture**: YOLOv8 or similar real-time object detection framework (adaptable)
- **Base Model**: Pre-trained backbone on COCO dataset with fine-tuning
- **Custom Layer**: Detection head optimized for tire-specific features
- **Training Split**: 80% Training Set, 20% Validation Set
- **Optimization**: Transfer learning on frozen backbone with trainable detection layers
- **Output**: `tire_count_model.h5` - production-ready detection model

### Step 6: Inference & Counting Pipeline
The deployed model:
- Accepts new tire photographs as input
- Runs real-time object detection to identify tire instances
- Outputs:
  - **Tire count**: Total number of detected tires
  - **Confidence scores**: Detection confidence for each identified tire
  - **Bounding coordinates**: Precise location of each tire in image space
  - **Visualization**: Annotated image with detection overlays

## Usage

### Loading and Inference

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('tire_count_model.h5')

# Preprocess input image
def preprocess_image(image_path, target_size=(416, 416)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Run inference
image_path = 'path/to/tire_image.jpg'
processed_image = preprocess_image(image_path)
predictions = model.predict(processed_image)

# Parse predictions to extract tire count
tire_count = len(predictions[0][predictions[0][:, 4] > 0.5])
print(f"Number of tires detected: {tire_count}")
```

### Batch Processing

```python
import os
from glob import glob

# Process multiple images
image_folder = './Count Tyres/'
results = {}

for image_file in glob(os.path.join(image_folder, '*.jpg')):
    processed = preprocess_image(image_file)
    predictions = model.predict(processed)
    tire_count = len(predictions[0][predictions[0][:, 4] > 0.5])
    results[image_file] = tire_count

# Display results
for image, count in results.items():
    print(f"{image}: {count} tires")
```

## Model Performance

- **Accuracy**: Evaluated on validation set with precision/recall metrics
- **Speed**: Real-time inference capable (processes images in <100ms)
- **Robustness**: Tested across varied lighting, angles, and occlusion scenarios

## Production Deployment

The model is optimized for:
- GPU-accelerated inference on NVIDIA hardware
- CPU-compatible inference for edge deployment
- Containerized deployment via Docker
- REST API integration for warehouse management systems

## Future Enhancements

- Multi-class detection (distinguish tire types during counting)
- 3D tire localization for robotic picking systems
- Real-time video stream processing
- Ensemble model approaches for improved accuracy
- Automated retraining pipeline with new data

## Contributing

To add new training data or improve model performance:
1. Capture new tire images in warehouse conditions
2. Annotate using Label Studio with polygon coordinates
3. Export annotations to CSV format
4. Integrate into training pipeline
5. Retrain model with expanded dataset

## References

- YOLO: Real-Time Object Detection - https://docs.ultralytics.com/
- Label Studio - https://labelstud.io/
- TensorFlow Object Detection - https://github.com/tensorflow/models/tree/master/research/object_detection
