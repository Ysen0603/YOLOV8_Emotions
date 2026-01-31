# YOLOV8 Emotions

## Description

This project focuses on real-time emotion detection using the YOLOv8 object detection model and OpenCV. It features a Graphical User Interface (GUI) built with Tkinter that allows users to analyze emotions from images and videos, providing personalized recommendations based on the detected mood.

## Project Structure

The project is organized as follows:

- **src/core**: Core logic scripts for processing images and videos.
- **src/gui**: Source code for the Tkinter-based user interface.
- **assets**: Contains static resources such as images, audio files, videos, and the detection models.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- ultralytics
- opencv-python
- numpy
- pillow

## Usage

To run the analysis on images via the GUI, execute the following command from the project root:

```bash
python src/gui/image_tkinter.py
```

To run the analysis on videos via the GUI:

```bash
python src/gui/video_tkinter.py
```
