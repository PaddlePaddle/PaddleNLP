[简体中文](label_studio_text.md) | English

# Label Studio User Guide - Text Classification

**Table of contents**

- [1. Installation](#1)
- [2. Text Classification Task Annotation](#2)
     - [2.1 Project Creation](#21)
     - [2.2 Data Upload](#22)
     - [2.3 Label Construction](#23)
     - [2.4 Task Annotation](#24)
     - [2.5 Data Export](#25)
     - [2.6 Data Conversion](#26)
     - [2.7 More Configuration](#27)

<a name="1"></a>

## 1. Installation

** Dependencies used in the following annotation examples:**

- Python 3.8+
- label-studio == 1.6.0

Use pip to install label-studio in the terminal:

```shell
pip install label-studio==1.6.0
```

Once the installation is complete, run the following command line:
```shell
label-studio start
```

Open [http://localhost:8080/](http://127.0.0.1:8080/) in the browser, enter the user name and password to log in, and start using label-studio for labeling.

<a name="2"></a>

## 2. Text Classification Task Annotation

<a name="21"></a>

#### 2.1 Project Creation

Click Create to start creating a new project, fill in the project name, description, and select ``Text Classification`` in ``Labeling Setup``.

- Fill in the project name, description

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/210772704-7d8ebe91-eeb7-4760-82ac-f3c6478b754b.png />
</div>

- Upload the txt format file locally, select ``List of tasks``, and then choose to import this project.

<a name="data"></a>

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/210775940-59809038-fa55-44cf-8c9d-1b19dcbdc8a6.png  />
</div>

- Define labels

<a name="label"></a>

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/210775986-6402db99-4ab5-4ef7-af8d-9a8c91e12d3e.png />
</div>

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/210776027-c4beb431-a450-43b9-ba06-1ee5455a95c5.png />
</div>

<a name="22"></a>

#### 2.2 Data Upload

You can continue to import local txt format data after project creation. See more details in [Project Creation](#data).

<a name="23"></a>

#### 2.3 Label Construction

After project creation, you can add/delete labels in Setting/Labeling Interface just as in [Project Creation](#label)

LabelStudio supports single-label data annotation by default. Modify the value of `choice` as `multiple` in the `code` tab when multiple-label annotation is required.

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/222630045-8d6eebf7-572f-43d2-b7a1-24bf21a47fad.png />
</div>

<a name="24"></a>

#### 2.4 Task annotation

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/210778977-842785fc-8dff-4065-81af-8216d3646f01.png />
</div>

<a name="25"></a>

#### 2.5 Data Export

Check the marked text ID, select the exported file type as ``JSON``, and export the data:

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/210779879-7560116b-22ab-433c-8123-43402659bf1a.png />
</div>

<a name="26"></a>

#### 2.6 Data conversion

First, create a label file in the `./data` directory, with one label candidate per line. You can also directly set label condidates list by `options`. Rename the exported file to ``label_studio.json`` and put it in the ``./data`` directory. Through the [label_studio.py](./label_studio.py) script, it can be converted to the data format of UTC.


```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --options ./data/label.txt
```

<a name="27"></a>

#### 2.7 More Configuration

- ``label_studio_file``: Data labeling file exported from label studio.
- ``save_dir``: The storage directory of the training data, which is stored in the ``data`` directory by default.
- ``splits``: The proportion of training set and validation set when dividing the data set. The default is [0.8, 0.1, 0.1], which means that the data is divided into training set, verification set and test set according to the ratio of ``8:1:1``.
- ``options``: Specify the label candidates set. For filename, there should be one label per line in the file. For list, the length should be longer than 1.
- ``is_shuffle``: Whether to randomly shuffle the data set, the default is True.
- ``seed``: random seed, default is 1000.

Note:
- By default the [label_studio.py](./label_studio.py) script will divide the data proportionally into train/dev/test datasets
- Each time the [label_studio.py](./label_studio.py) script is executed, the existing data file with the same name will be overwritten
- For files exported from label_studio, each piece of data in the default file is correctly labeled manually.

## References
- **[Label Studio](https://labelstud.io/)**
