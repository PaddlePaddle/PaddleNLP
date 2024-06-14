# Label Studio User Guide - Document Information Extraction

  **Table of contents**

- [1. Installation](#1)
- [2. Document Extraction Task Annotation](#2)
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

## 2. Document Extraction Task Annotation

<a name="21"></a>

#### 2.1 Project Creation

Click Create to start creating a new project, fill in the project name, description, and select ``Object Detection with Bounding Boxes``.

- Fill in the project name, description

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199445809-1206f887-2782-459e-9001-fbd790d59a5e.png height=300 width=1200 />
</div>

- For **Named Entity Recognition, Relation Extraction** tasks please select ``Object Detection with Bounding Boxes`

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199660090-d84901dd-001d-4620-bffa-0101a4ecd6e5.png height=400 width=1200 />
</div>

- For **Document Classification** task please select ``Image Classification`

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199729973-53a994d8-da71-4ab9-84f5-83297e19a7a1.png height=400 width=1200 />
</div>

- Define labels

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199450930-4c0cd189-6085-465a-aca0-6ba6f52a0c0d.png height=600 width=1200 />
</div>

The figure shows the construction of Span entity type tags. For the construction of other types of tags, please refer to [2.3 Label Construction](#23)

<a name="22"></a>

#### 2.2 Data upload

First upload the picture from a local or HTTP link, and then choose to import this project.

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199452007-2d45f7ba-c631-46b4-b21f-729a2ed652e9.png height=270 width=1200 />
</div>

<a name="23"></a>

#### 2.3 Label Construction

- Entity Label

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199456432-ce601ab0-7d6c-458f-ac46-8839dbc4d013.png height=500 width=1200 />
</div>


- Relation label

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199877621-f60e00c7-81ae-42e1-b498-8ebc5b5bd0fd.png height=650 width=1200 />
</div>

Relation XML template:

```xml
   <Relations>
     <Relation value="unit"/>
     <Relation value="Quantity"/>
     <Relation value="amount"/>
   </Relations>
```

- Classification label

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199891626-cc995783-18d2-41dc-88de-260b979edc56.png height=500 width=1200 />
</div>

<a name="24"></a>

#### 2.4 Task Annotation

- Entity extraction

     - Callout example:

         <div align="center">
             <img src=https://user-images.githubusercontent.com/40840292/199879427-82806ffc-dc60-4ec7-bda5-e16419ee9d15.png height=650 width=800 />
         </div>

     - The schema corresponding to this annotation example is:

         ```text
         schema = ['开票日期', '名称', '纳税人识别号', '地址、电话', '开户行及账号', '金额', '税额', '价税合计', 'No', '税率']
         ```

- Relation extraction

     - Step 1. Label the subject and object

        <div align="center">
            <img src=https://user-images.githubusercontent.com/40840292/218974459-4bf989fc-0e40-4dea-b309-346364cca1b5.png height=400 width=1000 />
        </div>

     - Step 2. Relation line, the direction of the arrow is from the subject to the object

        <div align="center">
            <img src=https://user-images.githubusercontent.com/40840292/218975474-0cf933bc-7c1e-4e7d-ada5-685ee5265f61.png height=450 width=1000 />
        </div>

        <div align="center">
            <img src=https://user-images.githubusercontent.com/40840292/218975743-dc718068-6d58-4352-8eb2-8973549dd971.png height=400 width=1000 />
        </div>

     - Step 3. Add corresponding relation label

        <div align="center">
            <img src=https://user-images.githubusercontent.com/40840292/218976095-ff5a84e8-302c-4789-98df-139a8cef8d5a.png height=360 width=1000 />
        </div>

        <div align="center">
            <img src=https://user-images.githubusercontent.com/40840292/218976368-a4556441-46ca-4372-b68b-e00b45f59260.png height=360 width=1000 />
        </div>

     - Step 4. Finish labeling

        <div align="center">
            <img src=https://user-images.githubusercontent.com/40840292/218976853-4903f2ec-b669-4c63-8c21-5f7184fc03db.png height=450 width=1000 />
        </div>


     - The schema corresponding to this annotation example is:

        ```text
        schema = {
            '名称及规格': [
                '金额',
                '单位',
                '数量'
            ]
        }
        ```

- Document classification

     - Callout example

         <div align="center">
             <img src=https://user-images.githubusercontent.com/40840292/199879238-b8b41d4a-7e77-47cd-8def-2fc8ba89442f.png height=650 width=800 />
         </div>

     - The schema corresponding to this annotation example is:

        ```text
        schema = '文档类别[发票，报关单]'
        ```


<a name="25"></a>

#### 2.5 Data Export

Check the marked image ID, select the exported file type as ``JSON``, and export the data:

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/199890897-b33ede99-97d8-4d44-877a-2518a87f8b67.png height=200 width=1200 />
</div>


<a name="26"></a>

#### 2.6 Data Conversion

After renaming the exported file to ``label_studio.json``, put it into the ``./document/data`` directory, and put the corresponding label image into the ``./document/data/images`` directory (The file name of the picture must be the same as the one uploaded to label studio). Through the [label_studio.py](./label_studio.py) script, it can be converted to the data format of UIE.

- Path example

```shell
./document/data/
├── images # image directory
│ ├── b0.jpg # Original picture (the file name must be the same as the one uploaded to label studio)
│ └── b1.jpg
└── label_studio.json # Annotation file exported from label studio
```

- Extraction task

```shell
python label_studio.py \
     --label_studio_file ./document/data/label_studio.json \
     --save_dir ./document/data \
     --splits 0.8 0.1 0.1 \
     --task_type ext
```

- Document classification tasks

```shell
python label_studio.py \
     --label_studio_file ./document/data/label_studio.json \
     --save_dir ./document/data \
     --splits 0.8 0.1 0.1 \
     --task_type cls \
     --prompt_prefix "document category" \
     --options "invoice" "customs declaration"
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
- ``separator``: The separator between entity category/evaluation dimension and classification label. This parameter is only valid for entity/evaluation dimension classification tasks. The default is"##".
- ``schema_lang``: Select the language of the schema, which will be the construction method of the training data prompt, optional `ch` and `en`. Defaults to `ch`.
- ``ocr_lang``: Select the language for OCR, optional `ch` and `en`. Defaults to `ch`.
- ``layout_analysis``: Whether to use PPStructure to analyze the layout of the document. This parameter is only valid for document type labeling tasks. The default is False.

Note:
- By default the [label_studio.py](./label_studio.py) script will divide the data proportionally into train/dev/test datasets
- Each time the [label_studio.py](./label_studio.py) script is executed, the existing data file with the same name will be overwritten
- In the model training phase, we recommend constructing some negative examples to improve the model performance, and we have built-in this function in the data conversion phase. The proportion of automatically constructed negative samples can be controlled by `negative_ratio`; the number of negative samples = negative_ratio * the number of positive samples.
- For files exported from label_studio, each piece of data in the default file is correctly labeled manually.


## References
- **[Label Studio](https://labelstud.io/)**
