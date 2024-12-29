# YOLO-NAS Car Detection

This repository contains a Python script to perform car detection using the YOLO-NAS model from SuperGradients. The model is trained to detect objects in images, specifically cars in this implementation.

## Requirements

- Python 3.x
- Google Colab (recommended for execution)

### Install Dependencies

To install the necessary dependencies, run the following command in your Colab notebook or terminal:

```bash
!pip install super-gradients opencv-python matplotlib
!pip install -U git+https://github.com/Deci-AI/super-gradients
```

## Description

This project utilizes the **YOLO-NAS model**, a state-of-the-art deep learning model for object detection, to detect cars in images. The script loads a pre-trained model, performs detection on an input image, and outputs the detected objects with bounding boxes drawn around them.

### Key Features:
- **Pre-trained YOLO-NAS Model**: Uses a pre-trained model for car detection.
- **Automatic URL Fix**: Automatically fixes an issue with the model checkpoint URL on Google Colab.
- **Object Detection**: Identifies cars in images and highlights them with bounding boxes.
- **Visualization**: Displays the result in a Jupyter notebook environment with the detected cars.

## How It Works

1. **Model Loading**: 
   The YOLO-NAS model is loaded from the SuperGradients library, with weights pretrained on the COCO dataset.
   
2. **Fixing URL Issue**: 
   A small modification is applied to the `checkpoint_utils.py` file to ensure that the model weights are correctly downloaded and handled in Google Colab.
   
3. **Car Detection**: 
   The script loads the image, applies the YOLO-NAS model to detect cars, and then draws bounding boxes around the detected cars.

4. **Result Display**: 
   After detection, the script saves the image with bounding boxes to the output directory and displays it in the notebook.

## Usage

### Step 1: Set Up Your Environment

In a Google Colab environment, simply copy the entire notebook or script and execute the cells. Ensure that the required dependencies are installed first.

### Step 2: Load Your Image

In the script, you can set the `image_path` variable to point to the location of the image you want to process. By default, it is set to `./test_images/car.jpg`.

### Step 3: Perform Object Detection

Run the script to load the YOLO-NAS model and detect cars in the image.

```python
# Define paths
image_path = "./test_images/car.jpg"  # Path to the test image
output_path = "./output"             # Output directory for results

# Load the YOLO-NAS model
model = load_yolo_nas_model("yolo_nas_s")

# Perform object detection
detect_objects(model, image_path, output_path)
```

### Step 4: View the Results

The output image with detected cars will be saved in the `./output` directory. It will also be displayed in the notebook.

## Code Explanation

- **`fix_checkpoint_utils()`**:  
   This function modifies the URL in the `checkpoint_utils.py` file to ensure the correct model weights are downloaded. It addresses a specific issue with Google Colab’s model downloading process.

- **`load_yolo_nas_model()`**:  
   Loads the YOLO-NAS model from the SuperGradients library with pretrained weights. The model type can be adjusted (e.g., "yolo_nas_s", "yolo_nas_m", "yolo_nas_l").

- **`detect_objects()`**:  
   This function performs object detection on the input image. It reads the image, runs the YOLO-NAS model on it, and extracts the bounding boxes and confidence scores for detected objects (specifically cars).

- **`cv2` and `matplotlib`**:  
   OpenCV (`cv2`) is used to handle image processing and drawing bounding boxes, while `matplotlib` is used to visualize the result.

## Contributing

If you would like to contribute to this project, please fork the repository, create a new branch, and submit a pull request with your changes. Contributions are always welcome!

## License

This project uses the YOLO-NAS model, which is licensed under [LICENSE.YOLONAS.md].  
See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

- SuperGradients library for providing the YOLO-NAS model.
- YOLO-NAS, Deci-AI for their outstanding work on the model.
