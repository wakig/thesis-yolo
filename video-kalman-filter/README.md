# YOLOv3-video-detection
Video inferencing of object detection using YOLOv3 (weights pre-trained on COCO dataset) along with tracking of a single object using a Kalman filter.

## How to Use
The YOLOv3 weights file was not included in this repository due to a large file size. <br>
Please download `yolov3.weights` from https://drive.google.com/drive/folders/1j6kixXJj-muUk14swugCrvrJF9xEsl09?usp=share_link and put it in the `yolo-coco-data` folder.

Next, put the video you want to test as `videos/test.mp4` (ideally containing a single person). Then run `yolov3-video-detection.py`. This outputs `videos/result.mp4` in which a Kalman filter is applied on a single 'person' object.

## References
https://github.com/nitish-gautam/YOLOv3-video-detection