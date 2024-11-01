# Label Studio User Guide - Text Information Extraction

**Table of contents**

- [1. Installation](#1)
- [2. Text Extraction Task Annotation](#2)
     - [2.1 Project Creation](#21)
     - [2.2 Data Upload](#22)
     - [2.3 Label Construction](#23)
     - [2.4 Task Annotation](#24)
     - [2.5 Data Export](#25)
     - [2.6 Data Conversion](#26)
     - [2.7 More Configuration](#27)

<a name="1"></a>

## 1. Installation

**Environmental configuration used in the following annotation examples:**

- Python 3.8+
- label-studio == 1.6.0
- paddleocr >= 2.6.0.1

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

## 2. Text extraction task annotation

<a name="21"></a>

#### 2.1 Project Creation

Click Create to start creating a new project, fill in the project name, description, and select ``Object Detection with Bounding Boxes``.

- Fill in the project name, description

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199661377-d9664165-61aa-4462-927d-225118b8535b.png height=230 width=1200 />
</div>

- For **Named Entity Recognition, Relation Extraction, Event Extraction, Opinion Extraction** tasks please select ``Relation Extraction`.

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199661638-48a870eb-a1df-4db5-82b9-bc8e985f5190.png height=350 width=1200 />
</div>

- For **Text classification, Sentence-level sentiment classification** tasks please select ``Text Classification``.

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/212617773-34534e68-4544-4b24-8f39-ae7f9573d397.png height=420 width=1200 />
</div>

- Define labels

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199662737-ed996a2c-7a24-4077-8a36-239c4bfb0a16.png height=380 width=1200 />
</div>

The figure shows the construction of entity type tags, and the construction of other types of tags can refer to [2.3 Label Construction](#23)

<a name="22"></a>

#### 2.2 Data upload

First upload the txt format file locally, select ``List of tasks``, and then choose to import this project.

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199667670-1b8f6755-b41f-41c4-8afc-06bb051690b6.png height=210 width=1200 />
</div>

<a name="23"></a>

#### 2.3 Label construction

- Entity label

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199667941-04e300c5-3cd7-4b8e-aaf5-561415414891.png height=480 width=1200 />
</div>

- Relation label

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199725229-f5e998bf-367c-4449-b83a-c799f1e3de00.png height=620 width=1200 />
</div>

Relation XML template:

```xml
   <Relations>
     <Relation value="Singer"/>
     <Relation value="Published"/>
     <Relation value="Album"/>
   </Relations>
```

- Classification label

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199724082-ee82dceb-dab0-496d-a930-a8ecb284d8b2.png height=370 width=1200 />
</div>


<a name="24"></a>

#### 2.4 Task annotation

- Entity extraction

Callout example:

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199879957-aeec9d17-d342-4ea0-a840-457b49f6066e.png height=140 width=1000 />
</div>

The schema corresponding to this annotation example is:

```text
schema = [
    '时间',
    '选手',
    '赛事名称',
    '得分'
]
```

- Relation extraction

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199879866-03c1ecac-1828-4f35-af70-9ae61701c303.png height=230 width=1200 />
</div>

For relation extraction, the type setting of P is very important, and the following principles need to be followed

"{P} of {S} is {O}" needs to be able to form a semantically reasonable phrase. For example, for a triple (S, father and son, O), there is no problem with the relation category being father and son. However, according to the current structure of the UIE relation type prompt, the expression "the father and son of S is O" is not very smooth, so it is better to change P to child, that is, "child of S is O". **A reasonable P type setting will significantly improve the zero-shot performance**.

The schema corresponding to this annotation example is:

```text
schema = {
    '作品名': [
        '歌手',
        '发行时间',
        '所属专辑'
    ]
}
```

- Event extraction

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199879776-75abbade-9bea-44dc-ac36-322fecdc03e0.png height=220 width=1200 />
</div>

The schema corresponding to this annotation example is:

```text
schema = {
    '地震触发词': [
        '时间',
        '震级'
    ]
}
```

- Sentence level classification

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199879672-c3f286fe-a217-4888-950f-d4ee45b19f5a.png height=210 width=1000 />
</div>


The schema corresponding to this annotation example is:

```text
schema = '情感倾向[正向，负向]'
```

- Opinion Extraction

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199879586-8c6e4826-a3b0-49e0-9920-98ca062dccff.png height=240 width=1200 />
</div>

The schema corresponding to this annotation example is:

```text
schema = {
    '评价维度': [
        '观点词',
        '情感倾向[正向，负向]'
    ]
}
```

<a name="25"></a>

#### 2.5 Data Export

Check the marked text ID, select the exported file type as ``JSON``, and export the data:

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199891344-023736e2-6f9d-454b-b72a-dec6689f8436.png height=180 width=1200 />
</div>

<a name="26"></a>

#### 2.6 Data conversion

Rename the exported file to ``label_studio.json`` and put it in the ``./data`` directory. Through the [label_studio.py](./label_studio.py) script, it can be converted to the data format of UIE.

- Extraction task

```shell
python label_studio.py\
     --label_studio_file ./data/label_studio.json \
     --save_dir ./data \
     --splits 0.8 0.1 0.1 \
     --task_type ext
```

- Sentence-level classification tasks

In the data conversion stage, we will automatically construct prompt information for model training. For example, in sentence-level sentiment classification, the prompt is ``Sentiment Classification [positive, negative]``, which can be configured through `prompt_prefix` and `options` parameters.

```shell
python label_studio.py\
     --label_studio_file ./data/label_studio.json \
     --task_type cls \
     --save_dir ./data \
     --splits 0.8 0.1 0.1 \
     --prompt_prefix "Sentiment Classification" \
     --options "positive" "negative"
```

- Opinion Extraction

In the data conversion stage, we will automatically construct prompt information for model training. For example, in the emotional classification of the evaluation dimension, the prompt is ``Sentiment Classification of xxx [positive, negative]``, which can be declared through the `prompt_prefix` and `options` parameters.

```shell
python label_studio.py\
     --label_studio_file ./data/label_studio.json \
     --task_type ext \
     --save_dir ./data \
     --splits 0.8 0.1 0.1 \
     --prompt_prefix "Sentiment Classification" \
     --options "positive" "negative" \
     --separator "##"
```

<a name="27"></a>

#### 2.7 More Configuration

- ``label_studio_file``: Data labeling file exported from label studio.
- ``save_dir``: The storage directory of the training data, which is stored in the ``data`` directory by default.
- ``negative_ratio``: The maximum negative ratio. This parameter is only valid for extraction tasks. Properly constructing negative examples can improve the model effect. The number of negative examples is related to the actual number of labels, the maximum number of negative examples = negative_ratio * number of positive examples. This parameter is only valid for the training set, and the default is 5. In order to ensure the accuracy of the evaluation indicators, the verification set and test set are constructed with all negative examples by default.
- ``splits``: The proportion of training set and validation set when dividing the data set. The default is [0.8, 0.1, 0.1], which means that the data is divided into training set, verification set and test set according to the ratio of ``8:1:1``.
- ``task_type``: Select the task type, there are two types of tasks: extraction and classification.
- ``options``: Specify the category label of the classification task, this parameter is only valid for the classification type task. Defaults to ["positive", "negative"].
- ``prompt_prefix``: Declare the prompt prefix information of the classification task, this parameter is only valid for the classification type task. Defaults to "Sentimental Tendency".
- ``is_shuffle``: Whether to randomly shuffle the data set, the default is True.
- ``seed``: random seed, default is 1000.
- ``schema_lang``: Select the language of the schema, which will be the construction method of the training data prompt, optional `ch` and `en`. Defaults to `ch`.
- ``separator``: The separator between entity category/evaluation dimension and classification label. This parameter is only valid for entity/evaluation dimension classification tasks. The default is"##".

Note:
- By default the [label_studio.py](./label_studio.py) script will divide the data proportionally into train/dev/test datasets
- Each time the [label_studio.py](./label_studio.py) script is executed, the existing data file with the same name will be overwritten
- In the model training phase, we recommend constructing some negative examples to improve the model performance, and we have built-in this function in the data conversion phase. The proportion of automatically constructed negative samples can be controlled by `negative_ratio`; the number of negative samples = negative_ratio * the number of positive samples.
- For files exported from label_studio, each piece of data in the default file is correctly labeled manually.


## References
- **[Label Studio](https://labelstud.io/)**
