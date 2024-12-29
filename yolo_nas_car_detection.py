import os
from super_gradients.training import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def fix_checkpoint_utils():
    """
    Fix the URL in the checkpoint_utils.py file before loading the model.
    This modifies the 'unique_filename' line to handle the correct URL.
    """
    # Path to checkpoint_utils.py in Google Colab
    checkpoint_utils_path = "/usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/checkpoint_utils.py"

    # Check if the file exists
    if os.path.exists(checkpoint_utils_path):
        with open(checkpoint_utils_path, "r") as file:
            lines = file.readlines()

        # Search for the URL line and fix it
        modified = False
        for i, line in enumerate(lines):
            if 'url.split("https://sghub.deci.ai/models/")' in line:
                lines[i] = line.replace(
                    'url.split("https://sghub.deci.ai/models/")',
                    'url.split("https://sg-hub-nv.s3.amazonaws.com/models/")'
                )
                modified = True
                break

        if modified:
            # Write the corrected lines back to the file
            with open(checkpoint_utils_path, "w") as file:
                file.writelines(lines)
            print("Fix applied successfully.")
        else:
            print("No fix needed in checkpoint_utils.py.")
    else:
        print(f"checkpoint_utils.py not found at {checkpoint_utils_path}")

def load_yolo_nas_model(model_type="yolo_nas_s"):
    """
    Load a YOLO-NAS model from the SuperGradients library.

    Args:
        model_type (str): Model type, e.g., "yolo_nas_s", "yolo_nas_m", "yolo_nas_l".

    Returns:
        model: Loaded YOLO-NAS model.
    """
    # Apply the fix before loading the model
    fix_checkpoint_utils()

    print(f"Loading YOLO-NAS model: {model_type}")
    # Load the YOLO-NAS model with pretrained weights
    model = models.get(model_type, pretrained_weights="coco")
    return model

def detect_objects(model, image_path, output_path="./output"):
    """
    Perform object detection on an image using the YOLO-NAS model.
    :param model: Loaded YOLO-NAS model.
    :param image_path: Path to the input image.
    :param output_path: Directory to save the detection result.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform detection
    print("Detecting objects...")
    predictions = model.predict(image_rgb)

    # Extract prediction details
    bboxes = predictions.prediction.bboxes_xyxy
    confidences = predictions.prediction.confidence
    labels = predictions.prediction.labels
    class_names = ('car')

    # Filter detections for "car" class (class index for "car" is 2)
    car_detections = [
        (bbox, confidence) for bbox, label, confidence in zip(bboxes, labels, confidences)
        if class_names == "car"
    ]

    # Draw bounding boxes on the image
    for bbox, confidence in car_detections:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, f"Car {confidence:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Save and display results
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"detected_{Path(image_path).name}"
    cv2.imwrite(str(output_file), image)
    print(f"Detection complete. Results saved to {output_file}")

    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()



if __name__ == "__main__":
    # Define paths
    image_path = "./test_images/car.jpg"  # Path to the test image
    output_path = "./output"             # Output directory for results

    # Load the YOLO-NAS model
    model = load_yolo_nas_model("yolo_nas_s")

    # Perform object detection
    detect_objects(model, image_path, output_path)