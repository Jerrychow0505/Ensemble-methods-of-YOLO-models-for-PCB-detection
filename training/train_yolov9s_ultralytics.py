"""
Train YOLOv9s (Small variant) using Ultralytics framework
"""
from pathlib import Path
from ultralytics import YOLO

def train_yolov9s_ultralytics():
    """Train YOLOv9s with Ultralytics"""

    project_root = Path(__file__).parent.absolute()
    data_yaml = project_root / 'datasets' / 'pcb-filtered-yolov8' / 'data.yaml'

    print("=" * 70)
    print("Training YOLOv9s (Small) with Ultralytics Framework")
    print("=" * 70)
    print(f"Data: {data_yaml}")
    print(f"Model: YOLOv9s (Small - faster, smaller than YOLOv9c)")
    print("=" * 70)
    print()

    # Load YOLOv9s model from Ultralytics
    # Options: yolov9t, yolov9s, yolov9m, yolov9c, yolov9e
    model = YOLO('yolov9s.pt')  # Using yolov9s (small/fast)

    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov9s-ultralytics',
        project=str(project_root / 'runs' / 'yolov9s_ultralytics'),
        patience=50,
        save=True,
        device='cpu',  # Change to 'cuda' or '0' if GPU available
        workers=8,
        cache=True,
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("YOLOv9s (Ultralytics) Training Complete!")
    print("=" * 70)
    print(f"Results saved to: runs/yolov9s_ultralytics/yolov9s-ultralytics")
    print("=" * 70)

    return results

if __name__ == '__main__':
    train_yolov9s_ultralytics()
