import json
import os
import cv2

image_path = '/data/hjw/datasets/coco/val2017'
json_file = "/data/hjw/datasets/coco/annotations/instances_val2017.json"  # 目标检测生成的文件
visual_output = '/data/hjw/code/openDet/pudet2/testimg-output/gt'
json_file = open(json_file)
infos = json.load(json_file)
images = infos["images"]
annos = infos["annotations"]
assert len(images) == len(images)
for i in range(len(images)):
    im_id = images[i]["id"]
    im_path = image_path + "/" + images[i]["file_name"]
    img = cv2.imread(im_path)
    for j in range(len(annos)):
        if annos[j]["image_id"] == im_id:
            x, y, w, h = annos[j]["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            x2, y2 = x + w, y + h
            # object_name = annos[j][""]
            img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
            img_name = visual_output + "/" + images[i]["file_name"]
            cv2.imwrite(img_name, img)
            # continue
    print(i)


