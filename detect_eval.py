from ultralytics import YOLO
from pathlib import Path

def evaluate_yolo_model(model_path, data_path):
    """
    Modified version of YOLO model evaluation code.
    - Computes Precision, Recall, mAP, and per-class AP.

    Args:
        model_path (str): Path to the trained YOLO model.
        data_path (str): Path to the test dataset yaml file.

    Returns:
        dict: Metrics such as Precision, Recall, mAP, and per-class AP.
    """
    model = YOLO(model_path)

    results = model.val(data=data_path)

    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "Precision": results.box.mp,
        "Recall": results.box.mr,
        "AP_per_class": results.box.maps
    }

    print(f"mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print("AP per class:", metrics["AP_per_class"])

    return metrics


model_path = "best.pt"  # Path to the trained YOLO model file
data_path = "data.yaml"  # Path to the test dataset (yaml format)

results = evaluate_yolo_model(model_path, data_path)
