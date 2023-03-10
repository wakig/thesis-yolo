# Description
Python implementation of YOLOv3 object detection. Only trained to detect persons.

## How to Use
Some files were not included in the repository due to large file size or number of files.
* From https://drive.google.com/drive/folders/1j6kixXJj-muUk14swugCrvrJF9xEsl09?usp=share_link:
  * Download `checkpoint.pth.tar` and put it in the root directory.
  * Download `labels.zip`, extract it, and put the `labels` folder in the `COCO` directory.
* Combine all images from COCO 2017 dataset (https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) into a new folder called `images`, and put it in the `COCO` directory.

To minimize installation problems, it is recommended to start from a clean Anaconda environment and install the required packages using Anaconda Navigator. (Note: Albumentations is included in the package named `imgaug`)

For training, run `train.py`. <br>
For inferencing, run `detect.py`. Currently, it only makes predictions (bounding boxes) on a few sample images from the test set.

## To Do
* Train the model further (and record metrics)
* Work on video/webcam inferencing

## References
https://github.com/SannaPersson/YOLOv3-PyTorch <br>
https://github.com/aladdinpersson/Machine-Learning-Collection