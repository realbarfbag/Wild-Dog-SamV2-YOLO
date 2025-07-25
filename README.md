# Wild-Dog-SamV2-YOLO
This repository implements a complete pipeline for detecting and segmenting **wild dogs** from camera trap footage using YOLOv8 and SAMv2. It supports both pretrained detection and custom YOLOv8 training using bounding box annotations.
![Image](https://github.com/user-attachments/assets/dc24347f-c907-4163-a8ec-c1bf209cf4fa)

Figure 1: Overview of the pipeline for Wild Dog Detection and Segmentation.

Installation Requirements

This pipeline requires torch, ultralytics, and dependencies for SAMv2.
The tools used in this process can be installed with:
<pre> ```
pip install torch torchvision
pip install ultralytics
pip install opencv-python matplotlib numpy
git clone https://github.com/realbarfbag/Wild-Dog-SamV2-YOLO.git
cd Wild-Dog-SamV2-YOLO
``` </pre>


![Image](https://github.com/user-attachments/assets/98996799-7324-4404-be01-94ab23eb373c)
