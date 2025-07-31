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


Step 1: Data Collection at The Wilds (2024)

The camera trap videos used in this pipeline were collected in Spring 2024 at The Wilds, a conservation facility in Ohio. Flights were conducted as part of a camera trap monitoring and behavioral observation effort targeting African painted dogs 
![Image](https://github.com/user-attachments/assets/98996799-7324-4404-be01-94ab23eb373c)
 Figure 2: Image of segmented and bounded dogs



Step 2: Preprocessing Video Data

The raw .mp4 video was processed as follows:
We used ffmpeg to extract frames from the video for processing:
<pre> ```
ffmpeg -i wild_dogs_video.mp4 -qscale:v 2 temp_frames_subset_128/%05d.jpg
``` </pre>
This creates a directory of 128 JPEG frames for downstream object detection.

Step 3: Generating json file for training

Because the video data contained only wild dogs, we were able to create a clean dataset focused on a single class: "dog".

Bounding boxes were generated via automatic detection and optional manual filtering. These detections were saved as a JSON file mapping filenames to [x1, y1, x2, y2] boxes.

The dataset was organized into YOLOv8’s expected directory structure:
<pre> ```
dog_yolo_dataset/
├── images/
│   ├── train/    ← 102 training images
│   └── val/      ← 26 validation images
└── labels/
    ├── train/    ← Corresponding .txt label files
    └── val/
``` </pre>
Each .txt file contains 1+ lines like: 0 0.375 0.375 0.25 0.25
Where:
    0 is the class ID for "dog"
    Remaining values are normalized center coordinates and box dimensions


Step 4: Wild Dog Detection using YOLOv8
To improve specificity, we trained a custom YOLOv8 model on our wild dog dataset. The annotation JSON file was converted to YOLO-format labels using:
<pre> ```
python convert_json_to_yolo.py \
    --json dog_boxes.json \
    --images_dir temp_frames_subset_128 \
    --output_dir dog_yolo_dataset
``` </pre>

The custom model was trained with the following command:
    <pre> ```
python convert_json_to_yolo.py \
    yolo detect train \
    data=dog_dataset.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640 \
    name=dog_train
``` </pre>

This generated a trained model saved as: runs/detect/dog_train/weights/best.pt

 Step 4: Segmentation with SAMv2
 Once bounding boxes were finalized, the SAMv2 (Segment Anything Model) was used to generate precise pixel-level masks of detected wild dogs:
 <pre> ```
python run_samv2.py \
    --image_dir temp_frames_subset_128 \
    --bbox_json dog_boxes.json \
    --sam_model sam2.1_b.pt \
    --output_dir sam_outputs_custom
``` </pre>
Outputs include:
    Segmented masks,
    Overlaid annotated frames,
    A compiled .mp4 video of results.
