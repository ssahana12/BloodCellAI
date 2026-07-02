import json
import os

classes = {
    "RBC": 0,
    "WBC": 1,
    "Platelets": 2
}

base = "/home/nvidia/BloodCellAI/datasets/bccd"

splits = ["train", "val", "test"]

for split in splits:
    ann_dir = f"{base}/{split}/ann"
    img_dir = f"{base}/{split}/img"

    out_dir = f"{base}/{split}/labels"
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(ann_dir):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(ann_dir, file)) as f:
            data = json.load(f)

        w = data["size"]["width"]
        h = data["size"]["height"]

        yolo_lines = []

        for obj in data["objects"]:
            label = obj["classTitle"]
            points = obj["points"]["exterior"]

            x1, y1 = points[0]
            x2, y2 = points[1]

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h

            cls = classes[label]
            yolo_lines.append(f"{cls} {xc} {yc} {bw} {bh}")

        out_file = file.replace(".json", ".txt")
        with open(os.path.join(out_dir, out_file), "w") as f:
            f.write("\n".join(yolo_lines))

print("Done converting BCCD → YOLO format")
