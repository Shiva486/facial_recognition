# facial_recognition
- Setup _opencv for python3_ on raspberry pi. Follow these instructions: https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/
- Install the following python packages:
```pip install numpy
pip install scipy
pip install scikit-image
pip install dlib
pip install ubidots==1.6.6
```
- Command for starting the application:<br/> 
``` 
export DISPLAY=:0
*python faceDetectionDNNMultiThread.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel*
```
