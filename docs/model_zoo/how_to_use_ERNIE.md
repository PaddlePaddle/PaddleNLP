# 在 PaddleNLP 中如何使用文心大模型

* [模型介绍（PaddleNLP中接入的文心大模型）](#模型介绍)
    * [开源开放的文心大模型](#开源开放的文心大模型)
    * [基于文心大模型的API服务](#基于文心大模型的API服务)
* [如何使用](#如何使用)
* [如何接入](#如何接入)

产业级知识增强大模型——文心大模型，目前已成为千行百业AI开发的首选基座大模型。作为自然语言处理模型库，PaddleNLP 持续开源了众多效果领先的文心大模型，覆盖各任务通用的预训练模型，以及面向特定任务的预训练模型，你可以直接调用此类模型，也可以进行模型二次开发与调优；除了开源开放的模型之外，为满足开发者需求，[PaddleNLP Pipelines](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines) 中也接入了基于文心大模型的 API 服务，作为 Pipelines 基础组件，供开发者灵活插拔，像搭积木一样组建自己的NLP产线系统。

## 模型介绍与使用

### 开源开放的文心大模型

<table style="width:100%;" cellpadding="2" cellspacing="0" border="1" bordercolor="#000000">
    <tbody>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;">模型名称</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">描述</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">参数量</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0">
                        ERNIE 3.0
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">各类 NLP 任务通用的预训练模型，在文心大模型 ERNIE 3.0 基础上通过在线蒸馏技术得到的一系列轻量级模型。</span>
            </td>
            <td style="text-align:center"><details><summary>size</summary><div>
                ERNIE 3.0-Base: 117.9 M</br>
                ERNIE 3.0-Medium: 75.4 M</br>
                ERNIE 3.0-Mini: 26.9 M	<br />
                ERNIE 3.0-Micro: 23.4 M<br />
                ERNIE 3.0-Nano: 17.9 M <br /> </div></details></td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie">
                        ERNIE-UIE
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">通用信息抽取模型，实体抽取、关系抽取、事件抽取、情感分析等多任务统一建模，零样本与小样本能力突出。</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-layout">
                        ERNIE-Layout
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">各类 NLP 任务通用，基于多语言跨模态布局增强的文档智能大模型，融合文本、图像、布局等信息进行跨模态联合建模，能够深度理解多模态文档，为各类上层应用提供SOTA模型底座。</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-m">
                        ERNIE-M
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">各类 NLP 任务通用的多语言预训练模型，能同时理解96种语言，在自然语言推断、语义检索、语义相似度、命名实体识别、阅读理解等5类典型跨语言理解任务上刷新世界最好效果。</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-health">
                        ERNIE-Health
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">医疗领域各类 NLP 任务通用的预训练模型，通过医疗知识增强技术进一步学习海量的医疗数据，精准地掌握了专业的医学知识。</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-gen">
                        ERNIE-Gen
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">面向生成任务的预训练-微调框架。</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-doc">
                        ERNIE-Doc
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">各类 NLP 任务通用，针对长文本的预训练模型。在循环Transformer机制之上，创新性地提出两阶段重复学习以及增强的循环机制，以此提高模型感受野，加强模型对长文本的理解能力。</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_vil">
                        ERNIE-ViL
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">业界首个融合场景图知识的多模态预训练模型，在包括视觉常识推理、视觉问答、引用表达式理解、跨模态图像检索、跨模态文本检索等 5 项典型多模态任务中刷新了世界最好效果。并在多模态领域权威榜单视觉常识推理任务（VCR）上登顶榜首。</span>
            </td>
       <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        ERNIE-Search
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">面向检索任务、效果领先的预训练模型，预训练阶段使用由细粒度交互向粗粒度交互蒸馏的策略，节省了传统方法中训练教师模型的开销，提高了模型效率（开源中）。</span>
            </td>
       <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        RocketQA
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">面向检索任务、效果领先的预训练模型（开源中）。</span>
            </td>
       <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/plato-xl">
                        PLATO-XL
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">面向对话任务，业界首个开源的百亿超大规模开放域对话预训练模型，模型参数量达11B量级，经过十亿级样本对话数据的预训练，并引入role embedding区分多方对话中的对话角色，效果领先。</span>
            </td>
    <tbody>
</table>
<br />

        
### 基于文心大模型的API服务

        
<table style="width:100%;" cellpadding="2" cellspacing="0" border="1" bordercolor="#000000">
    <tbody>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;">模型名称</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">功能描述</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/text_to_image">
                        ERNIE-ViLG
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">跨模态双向生成模型，通过自回归生成模式对图像生成和文本生成任务进行统一建模，图文双向生成效果领先。</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        ... ... 
                  </a> </span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">后续补充更多。</span>
            </td>
     <tbody>
</table>
<br />

## 如何使用
       
对于开源的预训练模型，你可以直接调用，也可以进行模型二次开发与调优。对于基于文心大模型的 API 服务，可使用 [PaddleNLP Pipelines](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines) 进行调用，API 服务作为 Pipelines 基础组件，供开发者灵活插拔，像搭积木一样组建自己的NLP产线系统。
       
<table style="width:100%;" cellpadding="2" cellspacing="0" border="1" bordercolor="#000000">
    <tbody>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;">模型名称</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">开源/ API？</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:18px;">支持二次开发</span>
            </td>
           <td style="text-align:center">
                <span style="font-size:18px;">支持 Taskflow 一键预测</span>
            </td>
           <td style="text-align:center">
                <span style="font-size:18px;">是否接入 Pipelines </span>
            </td>
           <td style="text-align:center">
                <span style="font-size:18px;">产业实践范例</span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0">
                        ERNIE 3.0
                  </a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://aistudio.baidu.com/aistudio/projectdetail/3996601">【快速上手ERNIE 3.0】法律文本多标签分类实战</a> </span><br />
                  <a href="https://aistudio.baidu.com/aistudio/projectdetail/3986803">【快速上手ERNIE 3.0】中文语义匹配实战</a> </span><br />
                  <a href="https://aistudio.baidu.com/aistudio/projectdetail/2017202">【快速上手ERNIE 3.0】对话意图识别</a> </span><br />
                  <a href="https://aistudio.baidu.com/aistudio/projectdetail/3955163">【快速上手ERNIE 3.0】中文情感分析</a> </span><br />
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0">
                        ERNIE-UIE
                  </a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">接入中</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://aistudio.baidu.com/aistudio/projectdetail/4038499">5条标注数据搞定快递单信息抽取 </a> </span>
            </td>
         <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0">
                        ERNIE-Layout
                  </a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/document-intelligence">
                        从零搭建端到端开放文档抽取问答系统
                  </a> </span>
            </td>
          <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0">
                        ERNIE-M
                  </a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        建设中
                  </a> </span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-health">ERNIE-Health</a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        建设中
                  </a> </span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-gen">ERNIE-Gen</a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        建设中
                  </a> </span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-doc">ERNIE-Doc</a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        建设中
                  </a> </span>
            </td>
       <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_vil">ERNIE-ViL</a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        建设中
                  </a> </span>
            </td>
        <tr>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/plato-xl">PLATO-XL</a> </span>
            </td>
            <td style="text-align:center"> <span style="font-size:18px;">开源</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">是</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center"> <span style="font-size:18px;">否</span></td>
            <td style="text-align:center;vertical-align:middle">
                <span style="font-size:18px;"> 
                  <a href="">
                        建设中
                  </a> </span>
            </td>
    <tbody>
</table>
<br />
       
## 如何接入

如果你希望使用 Pipelines 构建自己的NLP流水线系统，希望用到更多文心大模型 API，可以参考如下方式接入进来：

- [如何将文心大模型 API 服务接入 Pipelines](https://aistudio.baidu.com/aistudio/projectdetail/5011119)
       
- 参考：[ERNIE-ViLG 接入 Pipelines](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/pipelines/nodes/text_to_image_generator)
