"""
Train YOLOv10 on filtered PCB dataset
"""
from ultralytics import YOLO
from pathlib import Path

def train_yolov10():
    """Train YOLOv10 model"""

    # Get absolute paths
    project_root = Path(__file__).parent.absolute()

    # Use yolov8 dataset format (compatible with v10)
    data_yaml = project_root / 'datasets' / 'pcb-filtered-yolov8' / 'data.yaml'

    # Training parameters
    img_size = 640
    batch_size = 16
    epochs = 100
    model_name = 'yolov10s.pt'  # Small model with pre-trained weights

    print("=" * 60)
    print("Training YOLOv10 on Filtered PCB Dataset")
    print("=" * 60)
    print(f"Data YAML: {data_yaml}")
    print(f"Model: {model_name}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Classes: IC, Capacitor, Connector, Electrolytic Capacitor")
    print("=" * 60)

    # Load model
    model = YOLO(model_name)

    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project='runs/yolov10',
        name='pcb-filtered',
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        device='',  # Auto-select device
        workers=8,
        patience=100,
        cache=True
    )

    print("\n" + "=" * 60)
    print("YOLOv10 Training Complete!")
    print("=" * 60)

    return results

if __name__ == '__main__':
    train_yolov10()
