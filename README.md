# Traffic Sign Detection and Recognition (TSDR) Project

<video width="640" height="480" controls>
  <source src="./media/WhatsApp%20Video%202024-10-25%20at%2019.07.31.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

[test video](/media/WhatsApp%20Video%202024-10-25%20at%2019.07.31.mp4)

## Video Demo: https://youtu.be/n4YhIVxkOKM

## Overview

This project, part of the Digital Egypt Pioneers Initiative (DEPI), focuses on developing an AI-powered system for Traffic Sign Detection and Recognition (TSDR) using deep learning. The system leverages YOLO (You Only Look Once) object detection models to recognize traffic signs in real time, providing essential support for autonomous vehicles and driver assistance systems. This project also involved publishing a research paper, contributing to academic advancements in intelligent transportation.

## Methodologies

The project methodology includes:

- **Model Selection**: Multiple models (e.g., Faster R-CNN, YOLO variants) were trained and evaluated to balance speed and accuracy.
- **YOLOv10m**: Selected for its efficiency, the YOLOv10m model offers a strong combination of high accuracy and real-time processing capability.
- **Model Pruning**: We applied a 5% pruning technique to YOLOv10m, reducing computational demands while maintaining high accuracy and speed, optimizing it for deployment in resource-limited environments.

## Data Exploration

The dataset for traffic sign recognition includes 14 sign classes, such as stop signs, speed limits, and traffic lights. Data exploration steps included:

- **Class Distribution Analysis**: Ensuring a balanced distribution of signs across training, validation, and test sets.
- **Data Augmentation**: Techniques like rotation, scaling, and color adjustments were employed to improve model generalization across varied real-world conditions.

## Findings

![graphs](/media/image.png)
![conv matrix](/media/image-1.png)

The project achieved the following:

- **YOLOv10m Model**: Our final model achieved a precision of 95.95% and a recall of 89.5%, making it suitable for real-time applications.
- **Pruning Results**: The pruned YOLOv10m model maintained accuracy with reduced computational requirements, enabling efficient deployment.
- **Deployment**: The system was deployed using FastAPI with Docker for seamless web-based interaction, allowing real-time video and image detection.

## Getting Started

### Pull and Run Locally

You can pull and run this project locally using Docker or by setting up the environment with the required libraries.

### Prerequisites

- **Docker**: Make sure Docker is installed on your machine.
- **Python**: If running without Docker, ensure Python 3.x and pip are installed.

### Running with Docker

1. **Pull the Docker Image**:
    ```bash
    docker pull lebo2678/depi-yolo:v1
    ```
    **or you can build the image yourself using the Dockerfile**
    ```bash
    cd traffic-sign-detection-recognition
    docker build -t depi-yolo .
    ```

2. **Run the Docker Container**:
    ```bash
    docker run -p 8000:80 lebo2678/depi-yolo
    ```

3. **Access the Application**: Open your browser and go to `http://localhost:8000` to interact with the TSDR interface.

### Running Locally with Python Libraries

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Lebo2678/traffic-sign-detection-recognition.git
    cd traffic-sign-detection-recognition
    ```

2. **Install Required Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    uvicorn app.main:app --reload
    ```

4. **Access the Application**: Open your browser and go to `http://localhost:8000`.

## Contributing

Feel free to open issues or pull requests to suggest improvements or fixes. Your contributions are welcome!

## License

This project is licensed under the MIT License.
