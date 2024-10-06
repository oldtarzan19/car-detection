import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_model():
    """
    Load the pre-trained SSD MobileNet V2 model from TensorFlow Hub.
    """
    print("Loading model...")
    try:
        model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    return model

def load_image(image_path):
    """
    Load an image from disk and preprocess it for the model.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)  # Changed dtype to uint8
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    return img, img_tensor

def detect_cars(model, input_tensor, threshold=0.6):
    """
    Perform object detection and filter detections for cars.
    """
    detections = model(input_tensor)

    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_boxes = detections['detection_boxes'][0].numpy()

    # COCO dataset class ID for cars is 3
    CAR_CLASS_ID = 3

    # Filter detections for cars with confidence above the threshold
    car_indices = np.where((detection_classes == CAR_CLASS_ID) & (detection_scores > threshold))[0]
    car_boxes = detection_boxes[car_indices]
    car_scores = detection_scores[car_indices]

    return car_boxes, car_scores

def draw_boxes(image, boxes, scores, label="Car"):
    """
    Draw bounding boxes around detected cars.
    """
    height, width, _ = image.shape
    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (int(xmin * width), int(xmax * width),
                                      int(ymin * height), int(ymax * height))
        # Draw rectangle
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Put label with confidence score
        label_text = f"{label}: {int(score * 100)}%"
        cv2.putText(image, label_text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def display_image(image, title="Car Detection"):
    """
    Display the image using matplotlib.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Paths
    IMAGE_DIR = 'images'
    IMAGE_NAME = 'auto.jpg'  # Replace with your image name
    IMAGE_PATH = os.path.join(IMAGE_DIR, IMAGE_NAME)  # Replace with your image

    # Check if the image exists
    if not os.path.isfile(IMAGE_PATH):
        print(f"Image file '{IMAGE_NAME}' not found in '{IMAGE_DIR}' directory.")
        exit(1)

    # Load model
    model = load_model()

    # Load and preprocess image
    original_image, input_tensor = load_image(IMAGE_PATH)

    # Perform detection
    car_boxes, car_scores = detect_cars(model, input_tensor, threshold=0.5)

    print(f"Number of cars detected: {len(car_boxes)}")

    # Draw bounding boxes
    annotated_image = draw_boxes(original_image.copy(), car_boxes, car_scores)

    # Display the result
    display_image(annotated_image, title="Detected Cars")

    # Save the annotated image
    OUTPUT_NAME = 'annotated_image.jpg'
    OUTPUT_PATH = os.path.join(IMAGE_DIR, OUTPUT_NAME)
    cv2.imwrite(OUTPUT_PATH, annotated_image)
    print(f"Annotated image saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
