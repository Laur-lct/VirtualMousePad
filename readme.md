# VirtualMousePad
# Web camera based mouse control application based on CNN classifier.
## It allows to map user eye blinks to mouse button actions.

## This is a prototype demonstrating the idea. Classifier model trained on the [CEW dataset](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html) 

## Dependencies

* Python anaconda packages
* [Caffe](http://caffe.berkeleyvision.org/)
* [Dlib](http://dlib.net/)
* [OpenCV](http://opencv.org/)
* [PyUserInput](https://github.com/PyUserInput/PyUserInput)

## Getting Started
 Install all the dependencies. 
 You need a web camera that sees your face clearly (Just an ordinary laptop is OK) 
 For better results, ensure the user's face is lit evenly and webcam runs at at least 15 FPS.
 To start the applicaion execute "python ./app/main.py" from the console.
 You should see an info popup and a video preview afterwards.
 
 Sensitivity, mouse button mappings can be adjusted in code.
 
## Actions mapping
* Regular both eyes blink - left mouse button click
* Long both eyes blink - double click
* Right eye close/open - right mouse button down/up
* Left eye close/open - left mouse button down/up
* Double blink - middle mouse button clik
Mouse actions configured in mouseAndKeyboard.py 

## Contact & author
If you're interested in the project feel free to submit a pull request or contact me at Roman Semenyk <r.semenyk(at)gmail.com>

Code released under [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license.