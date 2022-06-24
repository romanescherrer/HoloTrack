## Citation

This work was presented at the 7th edition of the [CIBB international conference](https://davidechicco.github.io/cibb2021/index.html).

If you use this code for your research, please consider citing:
```
@inproceedings{inproceedings,
author = {Scherrer, Romane and Govan, Rodrigue and Quiniou, Thomas and Jauffrais, Thierry and Hugues, Lemonnier and Bonnet, Sophie and Selmaoui, Nazha},
year = {2021},
month = {11},
title = {Automatic Plankton Detection and Classification on Raw Hologram with a Single Deep Learning Architecture}
}
```

Download the paper [here](https://www.researchgate.net/publication/355926011_Automatic_Plankton_Detection_and_Classification_on_Raw_Hologram_with_a_Single_Deep_Learning_Architecture).


## Hologram simulation

The holograms are simulated from bright-field microscopy images that are labeled and saved as ROI. A sample of the dataset can be found in `dataset/train`.  The full dataset (In Situ Ichthyoplankton Imaging System) is open source and can be downloaded from [kaggle](https://www.kaggle.com/competitions/datasciencebowl/data).

To create a hologram dataset for an object detection task, run :
```bash
python create_dataset.py --data dataset/train --project Images --nb 50
```
where nb is the number of images to simulate. The script will use the images in `dataset/train` to simulate the transmission function and the corresponding holograms.

The holograms and the ground-truth files (yolov5 format) are created and saved in `Images/Holo/train`.

## Hologram detection
To train a model to detect the plankton on the raw holograms, clone the [yolov5 repo](https://github.com/ultralytics/yolov5)

A trained yolov5s model can be found in the folder  `models/Holo/small/`.

To compute the model performances (mAP,AP,...) we used the [Open-Source Visual Interface for Object Detection Metrics](https://github.com/rafaelpadilla/review_object_detection_metrics)


## Hologram Tracking
Note that we used torch.hub.load() to load the trained model, so you need Python>=3.7.0.


To create a hologram dataset for a Multi-object detection task, run :
```bash
python create_MOT_files.py --data dataset/train --mot-challenge-name Holo --nbframes 50 --nbsim 2 --weights Holotrack/models/Holo/small/best.pt
```
The script will use the images in `dataset/train` to simulate the plankton motion in a 2D channel. It does not save the videos, but stores the ground truth labels and tracking predictions in `MOT_data/` folder that can be used with [TrackEval](https://github.com/JonathonLuiten/TrackEval) that provides code for a number of different tracking evaluation metrics.  
