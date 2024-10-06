# Car Detection with TensorFlow and OpenCV

## Overview

This project leverages TensorFlow's pre-trained SSD MobileNet V2 model to detect cars in images. The script processes an input image, identifies cars within it, draws bounding boxes around detected cars, displays the result, and saves the annotated image for further use.

## Features

- **Object Detection**: Utilizes a state-of-the-art TensorFlow model to accurately detect cars in images.
- **Visualization**: Draws bounding boxes with confidence scores around detected cars for clear visualization.
- **Image Processing**: Handles image loading, preprocessing, and post-processing using OpenCV and TensorFlow.
- **Result Saving**: Saves the annotated image with detected cars for reference.

## Requirements

- **Python**: Version 3.6 or higher
- **Packages**:
  - `tensorflow`
  - `tensorflow_hub`
  - `numpy`
  - `opencv-python`
  - `matplotlib`

## Installation

Follow the steps below to set up the project environment and install necessary dependencies.

### 1. Clone the Repository

If you haven't already, clone the repository to your local machine:

```bash
git clone https://github.com/oldtarzan19/car-detection.git
cd car-detection
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies and avoid conflicts.

#### Using `venv` (Python 3)

```bash
python3 -m venv venv
```

This command creates a virtual environment named `venv` in your project directory.

### 3. Activate the Virtual Environment

Activate the virtual environment to ensure that all packages are installed in an isolated environment.

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **On macOS and Linux:**

  ```bash
  source venv/bin/activate
  ```

After activation, your terminal prompt will typically be prefixed with `(venv)` indicating that the virtual environment is active.

### 4. Install Required Packages

With the virtual environment activated, install the necessary Python packages using `pip`.

#### Using `requirements.txt` (Recommended)

1. **Create a `requirements.txt` File**

   Create a file named `requirements.txt` in the project root directory with the following content:

   ```txt
   tensorflow
   tensorflow_hub
   numpy
   opencv-python
   matplotlib
   ```

2. **Install Dependencies**

   Run the following command to install all dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

#### Installing Packages Individually

Alternatively, you can install each package separately:

```bash
pip install tensorflow tensorflow_hub numpy opencv-python matplotlib
```

*Note: Installing via `requirements.txt` is recommended for easier dependency management.*

## Usage

Follow these steps to run the car detection script.

### 1. Prepare the Image

1. **Create an `images` Directory**

   Ensure there's an `images` directory in the project root. If not, create one:

   ```bash
   mkdir images
   ```

2. **Add Your Image**

   Place the image you want to process inside the `images` directory. By default, the script looks for an image named `auto.jpg`. You can either:

   - Rename your image to `auto.jpg`, or
   - Modify the `IMAGE_NAME` variable in the script to match your image's filename.

   **Directory Structure:**

   ```
   car-detection/
   ├── images/
   │   └── auto.jpg
   ├── detect_cars.py
   └── README.md
   ```

### 2. Run the Script

With the virtual environment activated and dependencies installed, execute the script:

```bash
python detect_cars.py
```

**Script Breakdown:**

- **Model Loading**: Downloads and loads the SSD MobileNet V2 model from TensorFlow Hub.
- **Image Processing**: Reads and preprocesses the input image.
- **Car Detection**: Identifies cars in the image with a confidence threshold of 50%.
- **Visualization**: Draws bounding boxes around detected cars and displays the image.
- **Saving Results**: Saves the annotated image as `annotated_image.jpg` in the `images` directory.

### 3. View the Results

After running the script:

- **Display**: A window will pop up displaying the image with detected cars highlighted.
- **Annotated Image**: Check the `images` directory for `annotated_image.jpg`, which contains the image with bounding boxes around detected cars.
