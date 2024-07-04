## PUDet

> **Pseudo-unknown Uncertainty for Open Set Object Detection**<br>

PUDet: PUDet is implemented based on [detectron2](https://github.com/facebookresearch/detectron2) and Opendet

### Train and Test

* **Testing**

Then, run the following command:
```
python tools/train_net.py --num-gpus 4 --config-file configs/faster_rcnn_R_50_FPN_6x_pudet.yaml \
        --eval-only MODEL.WEIGHTS output/faster_rcnn_R_50_FPN_6x_pudet/model_final.pth
```

* **Training**

The training process is the same as detectron2.
```
python tools/train_net.py --num-gpus 4 --config-file configs/faster_rcnn_R_50_FPN_6x_pudet.yaml
```
