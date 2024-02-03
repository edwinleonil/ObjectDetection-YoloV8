# ============== IMPORTANT - USE YoloV8 environment ====================
# ============== IMPORTANT - USE YoloV8 environment ====================

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('runs/detect/train/weights/best.pt')  # load a partially trained model

    # Resume training - only when training hasn't completed the total epochs
    results = model.train(resume=True)