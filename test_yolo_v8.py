# ============== IMPORTANT - USE YoloV8 environment ====================
# ============== IMPORTANT - USE YoloV8 environment ====================

# use ultralytics package to detect objects in the people.jpg image

from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (recommended for training)
model = YOLO('models/yolov8n.pt')

# Run inference on a single image
results = model('people.jpg', save=False, show=True)

# wait for a key press to exit
cv2.waitKey(0)

