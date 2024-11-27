import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# Constants
IMG_WIDTH = 416
IMG_HEIGHT = 416
NUM_CATEGORIES = 14

# Define base directories and model path
BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL = YOLO(os.path.join(BASE_DIR, 'best.pt'))
# print(MODEL)


# Output directory for saving results
OUTPUT_DIR = os.path.join(BASE_DIR, 'static/outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names for predictions
names = {0: 'Green Light', 1: 'Red Light', 2: 'Speed Limit 100', 3: 'Speed Limit 110', 4: 'Speed Limit 120',
         5: 'Speed Limit 20', 6: 'Speed Limit 30', 7: 'Speed Limit 40', 8: 'Speed Limit 50',
         9: 'Speed Limit 60', 10: 'Speed Limit 70', 11: 'Speed Limit 80', 12: 'Speed Limit 90', 13: 'Stop'}

# Function to load and process an image


def load_img(img_file):
    # Load the image from the file path
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(
            "Image could not be loaded. Please check the file path.")

    # Resize the image to the required size for YOLO
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Run YOLO model prediction on the resized image
    results = MODEL(image_resized)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if results are returned and there are detected boxes
    if results and results[0].boxes:  # Make sure results and boxes are not empty
        # Draw predictions on the image
        image_with_predictions = results[0].plot()

        # Save the predicted image with bounding boxes
        output_file_path = os.path.join(OUTPUT_DIR, os.path.basename(img_file))
        cv2.imwrite(output_file_path, image_with_predictions)

        # Prepare results for return, including confidence and class IDs
        predictions = []
        for box in results[0].boxes:  # Loop through each box (detected object)
            class_id = names[int(box.cls.item())]  # Extract class ID
            confidence = float(box.conf.item())  # Extract confidence
            predictions.append(
                {"class_id": class_id, "confidence": confidence}
            )

        return {
            "out_path": output_file_path,
            "predictions": predictions
        }

    return {
        "out_path": None,
        "predictions": []
    }


def load_vid(vid_file):
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, os.path.basename(
        vid_file).replace('.mp4', '_output.webm'))

    video_capture = cv2.VideoCapture(vid_file)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file: {vid_file}")
        return None

    frame_count = 0
    max_confidence_predictions = {}

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (IMG_WIDTH, IMG_HEIGHT))

    while True:
        success, frame = video_capture.read()
        if not success:
            print(f"End of video or error reading frame {frame_count}.")
            break

        frame_out = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        if frame_count % 2 == 0:
            results = MODEL(frame_out)
            frame_out = results[0].plot()
            video_writer.write(frame_out)
            # video_writer.write(frame_out)
        # _, result_frame = cv2.imencode('.jpg', result_frame)

        for box in results[0].boxes:
            class_id = names[int(box.cls.item())]
            confidence = float(box.conf.item())
            if class_id not in max_confidence_predictions or confidence > max_confidence_predictions[class_id]:
                max_confidence_predictions[class_id] = confidence

        frame_count += 1
        print(f"Processed frame {frame_count}")

    video_capture.release()
    video_writer.release()

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Video saved successfully at: {output_file}")
    else:
        print("Error: Output video file was not created.")
        return None

    final_results = [{"class_id": cls, "confidence": conf}
                     for cls, conf in max_confidence_predictions.items()]

    return {
        "predictions": final_results,
        "out_path": output_file
    }
