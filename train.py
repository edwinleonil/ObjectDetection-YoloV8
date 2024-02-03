# ============== IMPORTANT - USE YoloV8 environment ====================
# ============== IMPORTANT - USE YoloV8 environment ====================

from ultralytics import YOLO

#TODO: Train the model including SurfaceMelt

if __name__ == '__main__':
    # Load a model
    model = YOLO('models/yolov8s.pt')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data='yolov8.yaml', epochs=200, patience=10, batch=32)