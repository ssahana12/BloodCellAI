from ultralytics import YOLO
import os
import sys
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "runs/detect/train-5/weights/best.pt"
OUTPUT_DIR = "reports"
CONF = 0.10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO(MODEL_PATH)


def generate_report(image_path, results):
    """Convert YOLO output into a clean lab-style report."""

    boxes = results[0].boxes
    names = results[0].names

    counts = {}

    if boxes is not None:
        for c in boxes.cls:
            label = names[int(c)]
            counts[label] = counts.get(label, 0) + 1

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []
    report.append("BLOOD CELL ANALYSIS REPORT")
    report.append("=" * 40)
    report.append(f"Image: {os.path.basename(image_path)}")
    report.append(f"Time: {report_time}")
    report.append("")

    report.append("CELL COUNTS:")
    if counts:
        for k, v in counts.items():
            report.append(f"  {k}: {v}")
    else:
        report.append("  No cells detected.")

    total = sum(counts.values())
    report.append("")
    report.append(f"Total Objects Detected: {total}")

    report.append("")
    report.append("INTERPRETATION:")

    rbc = counts.get("RBC", 0)
    wbc = counts.get("WBC", 0)

    if wbc > 3:
        report.append("  Elevated WBC detected. Possible infection signal.")
    else:
        report.append("  WBC level appears within normal range.")

    if rbc > 0:
        report.append("  RBCs detected in sample.")

    report.append("=" * 40)

    return "\n".join(report)


def run(image_path):
    """Run detection + generate report."""

    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return

    print("\nRunning detection...\n")

    results = model.predict(
    source=image_path,
    conf=CONF,
    imgsz=1280,
    save=True,
    device="cpu"
)

    report_text = generate_report(image_path, results)

    report_file = os.path.join(
        OUTPUT_DIR,
        f"report_{os.path.basename(image_path).split('.')[0]}.txt"
    )

    with open(report_file, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_file}")
    print(f"Output image saved in runs/detect/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 detect_and_report.py <image_path>")
        sys.exit()

    run(sys.argv[1])
