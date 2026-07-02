# Blood-Cell-AI

Blood Cell AI Dashboard is a computer vision project that uses a YOLOv8 object detection model to identify and count blood cells from microscope images. It detects three classes: RBC (Red Blood Cells), WBC (White Blood Cells), and Platelets, and generates a simple medical-style report with counts, graphs, and a downloadable PDF.

The project is built using a converted version of the BCCD (Blood Cell Count and Detection) dataset, where annotation files were converted into YOLO format for training.

The application is deployed as a Streamlit web dashboard, allowing users to upload microscope images (.jpg, .jpeg, .png) and receive real-time detection results.

**Example Output**

<img width="2421" height="1301" alt="image" src="https://github.com/user-attachments/assets/81d9ebde-5325-45bd-ac72-be4cd05c427f" />

**The Algorithm**

This project uses YOLOv8 (You Only Look Once version 8), a deep learning object detection algorithm that predicts bounding boxes and class probabilities in a single forward pass of a neural network.

How it works in this project:
Dataset Preparation
The original BCCD dataset contains microscope images with JSON annotations (Supervisely format).
Each annotation includes bounding boxes for:
RBC (Red Blood Cells)
WBC (White Blood Cells)
Platelets

A custom script converts these JSON annotations into YOLO format .txt files:

class_id x_center y_center width height
Dataset is split into:
train/
val/
test/

**Model Training**
A pretrained YOLOv8 model (yolov8n.pt) is fine-tuned on the dataset.
Training uses:
Image size: 512 or 640
Epochs: ~50
Batch size: 8
The model learns to detect blood cell shapes, textures, and densities.
Inference (Prediction)
The trained model (best.pt) is loaded into a Streamlit app.
When an image is uploaded:
YOLO detects bounding boxes
Each detection is classified as RBC, WBC, or Platelets
Results are counted and visualized

**Post-processing**
The app counts detected objects per class.
Generates:
Text report
Bar chart visualization
Annotated image output
PDF report using ReportLab

**Running this project**

_1. Install dependencies_

Make sure Python 3.8+ is installed.

Install required libraries:

pip install ultralytics streamlit opencv-python numpy pillow matplotlib reportlab

If running on Jetson or ARM device, install PyTorch compatible with your system from NVIDIA index if needed.

_2. Clone or navigate to project_
cd Blood-Cell-AI

_3. Ensure model exists_

Make sure your trained model exists at:

runs/detect/train-5/weights/best.pt

(or update the path inside app.py accordingly)

_4. Run Streamlit app_
streamlit run app.py

_5. Upload images_

Upload microscope images in:

.jpg
.jpeg
.png

**The model will:
**
Detect blood cells
Display bounding boxes
Count each class
Generate report + graph + PDF download

**Project Structure**
<img width="360" height="413" alt="image" src="https://github.com/user-attachments/assets/cb956c7e-70e5-4cd8-9a38-bbb925d7417b" />
