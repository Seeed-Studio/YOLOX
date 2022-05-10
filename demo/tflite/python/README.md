# Introduction
Translating the model in [yolox](https://github.com/Megvii-BaseDetection/YOLOX) to tensorflow2.0.  
you should download yolox model: [yolox_s](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth), 
[yolox_m](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth),
[yolox_l](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth),
[yolox_x](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth),
[yolox_nano](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth),
[yolox_tiny](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) 
in python folder( see the content ).   
Supported tensorflow: tensorflow saved_model, tflite
## Content
* tflite
  * python
      * models
        * tf_darknet.py
        * tf_network_blocks.py
        * tf_yolo_head.py
        * tf_yolo_pafpn.py
      * export.py
      * para.py
      * demo.py
      * README.md
      
**models**: model translation code with tensorflow.  
**tflite**: export code, save tf model and demo test, you need download 
[yolox model](https://github.com/Megvii-BaseDetection/YOLOX) to this folder. 

## Usage
### Step1: export tf model from yolox's path file.
    python path/to/export.py -n yolox_nano --tsize 640 --include saved_model --device cpu
                                yolox_tiny                       tflite               gpu
                                yolox_s
                                yolox_x
                                yolox_m
                                yolox_l

### Step2: run demo
This part is similar to torch's python demo, but be care that parameter **tsize** must be same as which set
in step1, and usage of parameter include is same to step1.  
You need put images or vedio in python forder, and run command as following:
```
python demo.py image -n yolox_tiny -c yolox_tiny_fp16.tflite --path dog.jpg --tsize 640 --conf 0.25 --nms 0.45 --save_result --include tflite 
```