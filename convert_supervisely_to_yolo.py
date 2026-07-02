import os
import json
import shutil

SOURCE = "/home/nvidia/Downloads/bccd"
DEST = "/home/nvidia/BloodCellAI/bccd_yolo"

classes = {
    "RBC": 0,
    "WBC": 1,
    "Platelets": 2
}

for split in ["train", "val", "test"]:

    img_src = os.path.join(SOURCE, split, "img")
    ann_src = os.path.join(SOURCE, split, "ann")

    img_dst = os.path.join(DEST, "images", split)
    lbl_dst = os.path.join(DEST, "labels", split)

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    for file in os.listdir(img_src):

        if not file.endswith(".jpeg"):
            continue

        shutil.copy(
            os.path.join(img_src, file),
            os.path.join(img_dst, file)
        )

        json_file = os.path.join(
            ann_src,
            file + ".json"
        )

        if not os.path.exists(json_file):
            continue

        with open(json_file) as f:
            data = json.load(f)

        width = data["size"]["width"]
        height = data["size"]["height"]

        label_path = os.path.join(
            lbl_dst,
            file.replace(".jpeg", ".txt")
        )

        with open(label_path, "w") as out:

            for obj in data["objects"]:

                cls = obj["classTitle"]

                if cls not in classes:
                    print("Skipping unknown class:", cls)
                    continue

                (x1, y1), (x2, y2) = obj["points"]["exterior"]

                xc = ((x1 + x2) / 2) / width
                yc = ((y1 + y2) / 2) / height
                w = abs(x2 - x1) / width
                h = abs(y2 - y1) / height

                out.write(
                    f"{classes[cls]} {xc} {yc} {w} {h}\n"
                )

print("Done!")
