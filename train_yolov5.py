# YOLOv5 Object Detection Training on Custom Data
# Based on: https://colab.research.google.com/github/roboflow/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb

# Install dependencies
!pip install roboflow
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install yolov5

# Download dataset from Roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="qb5zteXrTNTRqVwRO5N0")
project = rf.workspace("shelf-product-detection").project("shelf-product-detection-5panm")
version = project.version(4)
dataset = version.download("yolov5")

# Train YOLOv5 model
import yolov5

# Initialize YOLOv5 model
model = yolov5.load('yolov5s.pt')

# Train the model
results = model.train(
    data=dataset.location + '/data.yaml',
    epochs=100,
    imgsz=640,
    device=0,  # GPU device, use 'cpu' if no GPU available
    batch=16,
    patience=20,
    augment=True
)

# Validate the model
validation_results = model.val()

# Test on a sample image
predictions = model.predict(source='path/to/test/image.jpg', conf=0.5)

print("Training complete!")
print(f"Model saved at: {results.save_dir}")
