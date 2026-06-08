## Visualization

1. First, select the model and its corresponding configuration file. You can choose them from the
   [model zoo](model_zoo/dinov2) or [DINOv3 model zoo](model_zoo/dinov3).
2. Then, run the following command:

```bash
cd visualization

python video_demo.py \
  --config-file /path/to/config.yaml \
  --input /path/to/images_folder \
  --output /path/to/output \
  --opts MODEL.WEIGHTS /path/to/weight.pth
```

🔧 Replace `/path/to/config.yaml` with the path to the config file.  
🔧 Replace `/path/to/images_folder` with the path to the folder containing the video frames..  
🔧 Replace `/path/to/weight.pth` with the path to the checkpoint to evaluate.   
🔧 Replace `/path/to/output` with the path to the output folder.  

