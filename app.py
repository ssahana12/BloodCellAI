import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# ----------------------------
# LOAD MODEL (FORCE CPU SAFE)
# ----------------------------
MODEL_PATH = "runs/detect/train-5/weights/best.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="Blood Cell AI Dashboard", layout="wide")

st.title("Blood Cell AI Dashboard")
st.write("Upload a microscope image to analyze blood cells and generate reports.")

# ----------------------------
# UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ----------------------------
# REPORT FUNCTION
# ----------------------------
def generate_report_text(counts):
    lines = []
    lines.append("BLOOD CELL ANALYSIS REPORT")
    lines.append("=" * 40)

    total = sum(counts.values())
    lines.append(f"Total Cells Detected: {total}")
    lines.append("")

    for k, v in counts.items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("INTERPRETATION:")

    wbc = counts.get("WBC", 0)
    if wbc > 3:
        lines.append("Possible elevated WBC (infection signal)")
    else:
        lines.append("WBC within normal range")

    return "\n".join(lines)

# ----------------------------
# PDF FUNCTION (FIXED)
# ----------------------------
def create_pdf(report_text):
    file_path = "report.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)
    text = c.beginText(40, 750)

    for line in report_text.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.save()

    return file_path

# ----------------------------
# MAIN APP
# ----------------------------
if uploaded_file is not None:

    # Convert image properly for YOLO
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    st.write("Running AI detection...")

    # IMPORTANT FIX: pass numpy array, not PIL
    results = model.predict(
        source=img_array,
        conf=0.2,
        iou=0.7,
        imgsz=640,
        device="cpu",   # FORCE CPU (fix your Jetson CUDA issue)
        verbose=False
    )

    boxes = results[0].boxes
    names = results[0].names

    counts = {"RBC": 0, "WBC": 0, "Platelets": 0}

    if boxes is not None:
        for c in boxes.cls:
            label = names[int(c)]
            if label in counts:
                counts[label] += 1

    annotated = results[0].plot()

    with col2:
        st.subheader("Detected Output")
        st.image(annotated, use_container_width=True)

    # ----------------------------
    # REPORT
    # ----------------------------
    report_text = generate_report_text(counts)

    st.subheader("Analysis Report")
    st.text(report_text)

    # ----------------------------
    # GRAPH
    # ----------------------------
    st.subheader("Cell Distribution")

    if sum(counts.values()) > 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(counts.keys(), counts.values())
        ax.set_title("Detected Cell Distribution")
        st.pyplot(fig)

    # ----------------------------
    # PDF DOWNLOAD
    # ----------------------------
    pdf_file = create_pdf(report_text)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download PDF Report",
            data=f,
            file_name="blood_cell_report.pdf",
            mime="application/pdf"
        )

    st.success("Analysis complete.")
