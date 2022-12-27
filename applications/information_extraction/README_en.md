# Information Extraction Application

**Table of contents**
- [1. Introduction to Information Extraction Application](#1)
- [2. Technical Features](#2)
   - [2.1 Full Coverage of Information Extraction Scheme](#21)
   - [2.2 Powerful Model Base](#22)
   - [2.3 Industrial-level Full-process Solution](#23)
   - [2.4 Case Display](#24)
- [3. Quick Start](#3)
   - [3.1 Taskflow](#31)
   - [3.2 Text Information Extraction](#32)
   - [3.3 Document Information Extraction](#33)

<a name="1"></a>

## 1. Introduction to Information Extraction Application

The information extraction application has open sourced industrial-level solutions for a series of high-frequency scenarios of information extraction. It has **multi-domain, multi-task, and cross-modal capabilities**. The whole process is perfect, which can quickly realize the landing of information extraction products.

In layman's terms, information extraction is the process of extracting structured information from given input data such as text/pictures. In the process of implementing information extraction, it usually faces many challenges such as changing fields, diverse tasks, and scarce data. Aiming at the difficulties and pain points in the field of information extraction, the PaddleNLP information extraction application** is based on the idea of UIE unified modeling** and provides an industrial-level application solution for information extraction. **In addition to supporting plain text scene entities, relationships, events, opinions, etc. In addition to task extraction, it also supports end-to-end information extraction of documents/pictures/tables**. The application **does not limit the industry field and extraction target**, and can realize the seamless connection from the product prototype development, business POC stage to business landing and iteration stages, helping developers to achieve rapid adaptation and landing of extraction scenarios in specific fields.

**Information extraction application highlights:**

- **Comprehensive coverage of scenarios🎓:** Covers various mainstream tasks of information extraction, oriented to plain text and document scenarios, supports multiple languages, and meets developers' diverse information extraction needs.
- **Leading effect🏃:** Using the UIE series models with outstanding effects in plain text and multi-modality as the training base, it provides pre-trained models of various sizes to meet different needs, and has extensive and mature practical applicability .
- **Easy to use⚡:** Implementing three lines of code through Taskflow can realize quick calls without labeled data, and one line of commands can start information extraction training, easily complete deployment and go online, and lower the threshold for information extraction technology.
- **Efficient Tuning✊:** Developers can easily get started with the data labeling and model training process without any background knowledge of machine learning.

<a name="2"></a>

## 2. Technical Features

<a name="21"></a>

### 2.1 Full Coverage of Information Extraction Scheme

Multiple model selection, satisfying accuracy and speed, and adapting to different information extraction scenarios.

| Model Name | Usage Scenarios | Supporting Tasks |
| :------------------------------------------------ ----------: | :------------------------------------- -------------------- | :---------------------------- ----------------------- |
| `uie-base`<br />`uie-medium`<br />`uie-mini`<br />`uie-micro`<br />`uie-nano` | For **plain text** The **extractive** model of the scene supports **Chinese** | has the ability to extract general information such as entities, relationships, events, and comments |
| `uie-base-en` | An **extractive** model for **plain text** scenarios, supports **English** | has general information extraction capabilities such as entities, relationships, events, comments, etc. |
| `uie-m-base`<br />`uie-m-large` | An **extractive** model for **plain text** scenarios, supporting **Chinese and English** | with entities, relationships, General information extraction capabilities such as events and comments |
| <b>`uie-x-base`</b> | An **extractive** model for **plain text** and **document** scenarios, supports **Chinese and English** | supports plain text All the functions of the scene, and also support the end-to-end information extraction of documents/pictures/tables |

<a name="22"></a>

### 2.2 Powerful Model Base

The information extraction application uses the ERNIE 3.0 lightweight model as a pre-training model, and at the same time performs secondary pre-training on a large amount of information extraction data, so that the model can be adapted to a fixed prompt.

- Experimental results of Chinese text dataset

We conducted experiments on the self-built test sets of the three vertical texts of the Internet, medical care, and finance:

<table>
<tr><th row_span='2'><th colspan='2'>finance<th colspan='2'>healthcare<th colspan='2'>internet
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-base (12L768H)<td>46.43<td>70.92<td><b>71.83</b><td>85.72<td>78.33<td>81.86
<tr><td>uie-medium (6L768H)<td>41.11<td>64.53<td>65.40<td>75.72<td>78.32<td>79.68
<tr><td>uie-mini (6L384H)<td>37.04<td>64.65<td>60.50<td>78.36<td>72.09<td>76.38
<tr><td>uie-micro (4L384H)<td>37.53<td>62.11<td>57.04<td>75.92<td>66.00<td>70.22
<tr><td>uie-nano (4L312H)<td>38.94<td>66.83<td>48.29<td>76.74<td>62.86<td>72.35
<tr><td>uie-m-large (24L1024H)<td><b>49.35</b><td><b>74.55</b><td>70.50<td><b>92.66</b ><td>78.49<td><b>83.02</b>
<tr><td>uie-m-base (12L768H)<td>38.46<td>74.31<td>63.37<td>87.32<td>76.27<td>80.13
<tr><td>🧾🎓<b>uie-x-base (12L768H)</b><td>48.84<td>73.87<td>65.60<td>88.81<td><b>79.36</b> <td>81.65
</table>

0-shot means that no training data is directly used for prediction through ```paddlenlp.Taskflow```, and 5-shot means that each category contains 5 pieces of labeled data for model fine-tuning. **Experiments show that UIE can further improve the performance with a small amount of data (few-shot) in vertical scenes**.

- Experimental results of multimodal datasets

We experimented on the zero-sample effect of UIE-X on the self-built multi-modal test sets of three major scenarios: general, financial, and medical:

<table>
<tr><th ><th>General <th>Financial<th colspan='2'>Medical
<tr><td>🧾🎓<b>uie-x-base (12L768H)</b><td>65.03<td>73.51<td>84.24
</table>

The general test set contains complex samples from different fields and is the most difficult task.

<a name="23"></a>

### 2.3 Industrial-level Full-process Solution

**Research stage**

- At this stage, the target requirements are open and there is a lack of data accumulation. We provide a simple way of invoking Taskflow with three lines of code, which can quickly verify the effect in business scenarios without labeling data.
   - [Text Extraction Taskflow User Guide](./taskflow_text_en.md)
   - [Document Extraction Taskflow User Guide](./taskflow_doc_en.md)

**Data preparation stage**

- We recommend customizing your own information extraction model in actual business scenarios. We provide Label Studio labeling solutions for different extraction scenarios. Based on this solution, the seamless connection from data labeling to training data construction can be realized, which greatly reduces the time cost of data labeling and model customization.
   - [Text Extraction Labeling Guide](./label_studio_text_en.md)
   - [Document Extraction and Labeling Guide](./label_studio_doc_en.md).

**Model fine-tuning and closed domain distillation**

- Based on UIE's excellent small-sample fine-tuning capabilities, it realizes low-cost model customization and adaptation. At the same time, it provides an acceleration solution for closed domain distillation to solve the problem of slow extraction speed.
   - [Example of the whole process of text information extraction](./text/README_en.md)
   - [Example of document information extraction process](./document/README_en.md)

**Model Deployment**

- Provide an HTTP deployment solution to quickly implement the deployment and launch of customized models.
   - [Text Extract HTTP Deployment Guide](./text/deploy/simple_serving/README_en.md)
   - [Document Extract HTTP Deployment Guide](./document/deploy/simple_serving/README_en.md)

<a name="24"></a>

### 2.4 Case Display

- 🧾Experience the UIE-X function through the [Huggingface website](https://huggingface.co/spaces/PaddlePaddle/UIE-X):

<div align="center">
     <img src=https://user-images.githubusercontent.com/40840292/207856955-a01cd5dd-fd5c-48ae-b8fd-c69512a88845.png height=500 width=900 hspace='10'/>
</div>

- UIE-X end-to-end document extraction industry application example

   - Customs declaration

     <div align="center">
         <img src=https://user-images.githubusercontent.com/40840292/205879840-239ada90-1692-40e4-a17f-c5e963fdd204.png height=800 width=500 />
     </div>

   - Delivery Note (Need fine-tuning)

     <div align="center">
         <img src=https://user-images.githubusercontent.com/40840292/205922422-f2615050-83cb-4bf5-8887-461f5633e85c.png height=250 width=700 />
     </div>

   - VAT invoice (need fine-tuning)

     <div align="center">
         <img src=https://user-images.githubusercontent.com/40840292/206084942-44ba477c-9244-4ce2-bbb5-ba430c9b926e.png height=550 width=700 />
     </div>

  - Form (need fine-tuning)

    <div align="center">
        <img src=https://user-images.githubusercontent.com/40840292/207856330-7aa0d158-47e0-477f-a88f-e23a040504a3.png height=400 width=700 />
    </div>

<a name="3"></a>

## 3. Quick Start

<a name="31"></a>

### 3.1 Taskflow

- Out of the box with Taskflow
   👉 [Text Extraction Taskflow User Guide](./taskflow_text_en.md)
   👉 [Document Extraction Taskflow User Guide](./taskflow_doc_en.md)

<a name="32"></a>

### 3.2 Text Information Extraction

- Quickly start text information extraction 👉 [Text Information Extraction Guide](./text/README_en.md)

<a name="33"></a>

### 3.3 Document Information Extraction

- Quickly open document information extraction 👉 [Document Information Extraction Guide](./document/README_en.md)
