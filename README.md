# AUIMechatronicsCV
This repository serves as a reference point for students regarding the projects have done in the Computer Vision workshop by the AUI Mechatronics Club.

## Set up
In order to set up your machine for this workshop, you need to meet the following requirements.

1. Python 3.11.9 or lower
2. Pip with Version 20.3 or higher
3. Libraries (latest version unless it's specified)
    1. OpenCV
    2. Numpy
    3. Mediapipe
    4. TensorFlow with Version 2.10 or lower

Please follow the following instructions to meet these requirements, which are needed to run the code used in this workshop

### Python & Pip
You can check if your Python version is 3.11.9 or lower by running this command in your terminal.
For Windows, `python --version`. For MacOs/Linux, `python3 --version`.

In order to download Python 3.11.9. [click on this link](https://www.python.org/downloads/release/python-3119/)

Also, you can check if your Pip version is 20.3 or higher by running this command in your terminal. For Windows, `python -m pip --version`. For MacOs/Linux, `python3 -m pip --version`.

You can upgrade Pip to the latest version by running this command: For Windows, `pip install --upgrade pip`. For MacOS/Linux, `pip3 install --upgrade pip`

### OpenCV, Numpy, and MediaPipe Libraries
To install the latest version of these 3 libraries, you can run the following commands in your terminal.

#### For Windows
```
pip install opencv-contrib-python
pip install numpy
pip install mediapipe
```

#### For Unix-Based Systems (i.e Linux & MacOS)
```
pip3 install opencv-contrib-python
pip3 install numpy
pip3 install mediapipe
```

### TensorFlow
The installation of TensorFlow is a little complicated than the other libraries, especially for Windows users, because of the various dependencies are being involved. However, if you carefully followed the following instruction and correctly did the previous ones, you will be able to install TensorFlow without any error messages.

#### For Windows
##### Step 1: Install Microsoft Visual C++ Redistributable
1. Go to [the Microsoft Visual C++ downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
2. Scroll down the page to the Visual Studio 2015, 2017, 2019, and 2022 section.
3. Download and install the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, 2019, and 2022 **for your platform** (64-bit, arm architecture, etc).

##### Step 2 (Optional): Install GPU Drivers
If you have NVIDIA graphics card, and you want to make use of its computing power while using TensorFlow, you will need to install NVIDIA GPU driver if you haven't did already.

[Link to download NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx).

##### Step 3: Install TensorFlow
Finally, run the following command to install TensorFlow with a version of 2.10 or lower.
```
pip install "tensorflow<2.11"
```

###### Step 4 (Optional): Verify the installation:
If you want to check if TensorFlow has been installed successfully and works properly by running the following command in your terminal; You should expect something being printed in the terminal.
```
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

#### For Unix-Based Systems (i.e Linux & MacOS)
Run the following command to install TensorFlow:
```
python3 -m pip install tensorflow
```

You check if TensorFlow has been installed successfully and works properly by running the following command in your terminal; You should expect something being printed in the terminal.
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

# References:
https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_complete_label_map.pbtxt
https://drive.google.com/drive/folders/1GrFlJNaQ9eAKcFo9MdBa7lgC0xxl-PRw
https://github.com/AchrafHsain7/AUIMechatronicsCV