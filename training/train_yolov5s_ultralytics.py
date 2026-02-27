"""
Train YOLOv5s using Ultralytics framework
This will allow us to get full PR curve data during validation
"""
from ultralytics import YOLO
from pathlib import Path

def main():
    # Setup paths
    project_root = Path(__file__).parent.absolute()
    data_yaml = project_root / 'datasets' / 'pcb-filtered-yolov8' / 'data.yaml'

    print("=" * 70)
    print("Training YOLOv5s with Ultralytics")
    print("=" * 70)
    print()

    # Load YOLOv5s model from Ultralytics
    model = YOLO('yolov5s.pt')

    print(f"Model: YOLOv5s")
    print(f"Dataset: {data_yaml}")
    print()

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov5s-ultralytics',
        project='runs/yolov5_ultralytics',
        patience=50,
        save=True,
        plots=True,
        device='cpu',
        verbose=True,
        val=True
    )

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Results saved to: {results.save_dir}")

    # Run final validation
    print()
    print("Running final validation...")
    val_results = model.val(data=str(data_yaml))
    print(f"Final mAP@0.5: {val_results.box.map50:.4f}")
    print(f"Final mAP@0.5:0.95: {val_results.box.map:.4f}")

if __name__ == '__main__':
    main()
