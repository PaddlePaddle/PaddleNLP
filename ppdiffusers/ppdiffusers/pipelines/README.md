# PPDiffusers Pipelines

Pipelines提供了一种对各种SOTA扩散模型进行各种下游任务推理的简单方式。
大多数扩散模型系统由多个独立训练的模型和高度自适应的调度器(scheduler)组成，通过pipeline我们可以很方便的对这些扩散模型系统进行端到端的推理。

举例来说， [Stable Diffusion](https://huggingface.co/blog/stable_diffusion)由以下组件构成:
- Autoencoder
- Conditional Unet
- CLIP text encoder
- scheduler
- CLIPFeatureExtractor
- safety checker

这些组件之间是独立训练或创建的，同时在Stable Diffusion的推理运行中也是必需的，我们可以通过pipelines来对整个系统进行封装，从而提供一个简洁的推理接口。

我们通过pipelines在统一的API下提供所有开源且SOTA的扩散模型系统的推理能力。具体来说，我们的pipelines能够提供以下功能：
1. 可以加载官方发布的权重，并根据相应的论文复现出与原始实现相同的输出
2. 提供一个简单的用户界面来推理运行扩散模型系统，参见[Pipelines API](#pipelines-api)部分
3. 提供易于理解的代码实现，可以与官方文档一起阅读，参见[Pipelines汇总](#Pipelines汇总)部分
4. 可以很容易地由社区贡献

**【注意】** Pipelines不（也不应该）提供任何训练功能。
如果您正在寻找训练的相关示例，请查看[examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples).

## Pipelines汇总

下表总结了所有支持的Pipelines，以及相应的论文、任务、推理脚本。

| Pipeline                                                                                                                      | Source                                                                                                                       | Tasks | Inference
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|:---:|:---:|
| [alt_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/alt_diffusion)                 | [**Alt Diffusion**](https://arxiv.org/abs/2211.06679)   | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-alt_diffusion.py)
| [alt_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/alt_diffusion)                 | [**Alt Diffusion**](https://arxiv.org/abs/2211.06679)   | *Image-to-Image Text-Guided Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-alt_diffusion.py)
| [repaint](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/repaint)                 | [**Repaint**](https://arxiv.org/abs/2201.09865)                                                      | *Image Inpainting* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_inpainting-repaint.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Text-to-Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-stable_diffusion_2.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)               | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Image-to-Image Text-Guided Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-stable_diffusion_2.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                 | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Text-Guided Image Inpainting* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_image_inpainting-stable_diffusion_2.py)
| [stable_diffusion_2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                 | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release)                                            | *Text-Guided Image Upscaling* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_image_upscaling-stable_diffusion_2.py)
| [stable_diffusion_safe](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion_safe)                 | [**Safe Stable Diffusion**](https://arxiv.org/abs/2211.05105)                                                      | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-stable_diffusion_safe.py)
| [versatile_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/versatile_diffusion)                 | [**Versatile Diffusion**](https://arxiv.org/abs/2211.08332)                                                      | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-versatile_diffusion.py)
| [versatile_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/versatile_diffusion)                 | [**Versatile Diffusion**](https://arxiv.org/abs/2211.08332)                                                      | *Image Variation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_variation-versatile_diffusion.py)
| [versatile_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/versatile_diffusion)                 | [**Versatile Diffusion**](https://arxiv.org/abs/2211.08332)                                                      | *Dual Text and Image Guided Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/dual_text_and_image_guided_generation-versatile_diffusion.py)
| [vq_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/vq_diffusion)                 | [**VQ Diffusion**](https://arxiv.org/abs/2111.14822)                                                      | *Text-to-Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-vq_diffusion.py)
| [dance_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/dance_diffusion)                 | [**Dance Diffusion**](https://github.com/Harmonai-org/sample-generator)                                                      | *Unconditional Audio Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_audio_generation-dance_diffusion.py)
| [ddpm](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/ddpm)                                       | [**Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2006.11239)                                             | *Unconditional Image Generation* |  [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-ddpm.py)
| [ddim](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/ddim)                                       | [**Denoising Diffusion Implicit Models**](https://arxiv.org/abs/2010.02502)                                                  | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-ddim.py)
| [latent_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/latent_diffusion)               | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)                         | *Text-to-Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-latent_diffusion.py)
| [latent_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/latent_diffusion)               | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)                         | *Super Superresolution* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/super_resolution-latent_diffusion.py)
| [latent_diffusion_uncond](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/latent_diffusion_uncond) | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)                         | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-latent_diffusion_uncond.py)
| [pndm](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/pndm)                                       | [**Pseudo Numerical Methods for Diffusion Models on Manifolds**](https://arxiv.org/abs/2202.09778)                           | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-pndm.py)
| [score_sde_ve](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/score_sde_ve)                       | [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://openreview.net/forum?id=PxTIG12RRHS) | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-score_sde_ve.py)
| [score_sde_vp](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/score_sde_vp)                       | [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://openreview.net/forum?id=PxTIG12RRHS) | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-score_sde_vp.py)
| [stable_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Text-to-Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_to_image_generation-stable_diffusion.py)
| [stable_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)               | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Image-to-Image Text-Guided Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/image_to_image_text_guided_generation-stable_diffusion.py)
| [stable_diffusion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion)                 | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Text-Guided Image Inpainting* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/text_guided_image_inpainting-stable_diffusion.py)
| [stochastic_karras_ve](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/ppdiffusers/pipelines/stochastic_karras_ve)       | [**Elucidating the Design Space of Diffusion-Based Generative Models**](https://arxiv.org/abs/2206.00364)                    | *Unconditional Image Generation* | [link](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference/unconditional_image_generation-stochastic_karras_ve.py)

**【注意】** Pipelines可以端到端的展示相应论文中描述的扩散模型系统。然而，大多数Pipelines可以使用不同的调度器组件，甚至不同的模型组件。

## Pipelines API

扩散模型系统通常由多个独立训练的模型以及调度器等其他组件构成。
其中每个模型都是在不同的任务上独立训练的，调度器可以很容易地进行替换。
然而，在推理过程中，我们希望能够轻松地加载所有组件并在推理中使用它们，即使某个组件来自不同的库, 为此，所有pipeline都提供以下功能：


- `from_pretrained` 该方法接收PaddleNLP模型库id（例如`runwayml/stable-diffusion-v1-5`）或本地目录路径。为了能够准确加载相应的模型和组件，相应目录下必须提供`model_index.json`文件。

- `save_pretrained` 该方法接受一个本地目录路径，Pipelines的所有模型或组件都将被保存到该目录下。对于每个模型或组件，都会在给定目录下创建一个子文件夹。同时`model_index.json`文件将会创建在本地目录路径的根目录下，以便可以再次从本地路径实例化整个Pipelines。

- `__call__` Pipelines在推理时将调用该方法。该方法定义了Pipelines的推理逻辑，它应该包括预处理、张量在不同模型之间的前向传播、后处理等整个推理流程。
