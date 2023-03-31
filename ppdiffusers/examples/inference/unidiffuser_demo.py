# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from ppdiffusers import (
    CaptionDecoder,
    FrozenAutoencoderKL,
    FrozenCLIPEmbedder,
    UniDiffuserTextToImagePipeline,
    UViT,
)

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserTextToImagePipeline(
    clip_text_model=FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", max_length=77),
    unet=UViT(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
        pretrained_path="models/uvit_v1.pdparams",
    ),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64),
    vae=FrozenAutoencoderKL(
        pretrained_path="models/autoencoder_kl.pdparams"
    ),  # .from_pretrained('models/autoencoder_kl.pdparams'),
    scheduler=None,
)  # .from_pretrained("thu-ml/unidiffuser")
# pipe.remove_unused_weights()
pipe.clip_text_model = FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", max_length=77)
pipe.vae = FrozenAutoencoderKL(pretrained_path="models/autoencoder_kl.pdparams")
pipe.caption_decoder = CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64)
prompt = "an child riding on a dog on the moon"
image = pipe(prompt=prompt, generator=generator).images[0]
image.save("./unidiffuser-t2i.png")


import paddle

from paddlenlp.transformers import CLIPModel, CLIPProcessor
from ppdiffusers import (
    CaptionDecoder,
    FrozenAutoencoderKL,
    UniDiffuserImageToTextPipeline,
    UViT,
)

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserImageToTextPipeline(
    image_encoder=CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    image_feature_extractor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    unet=UViT(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
        pretrained_path="models/uvit_v1.pdparams",
    ),
    vae=FrozenAutoencoderKL(
        pretrained_path="models/autoencoder_kl.pdparams"
    ),  # .from_pretrained('models/autoencoder_kl.pdparams'),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64),
    scheduler=None,
)  # .from_pretrained("thu-ml/unidiffuser")
# pipe.remove_unused_weights()
pipe.vae = FrozenAutoencoderKL(pretrained_path="models/autoencoder_kl.pdparams")
pipe.caption_decoder = CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64)
image = "./unidiffuser-i.png"
text = pipe(image=image, generator=generator).texts[0]
with open("./unidiffuser-i2t.txt", "w") as f:
    print("{}\n".format(text), file=f)


import paddle

from paddlenlp.transformers import CLIPModel, CLIPProcessor
from ppdiffusers import (
    CaptionDecoder,
    FrozenAutoencoderKL,
    UniDiffuserImageVariationPipeline,
    UViT,
)

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserImageVariationPipeline(
    image_encoder=CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    image_feature_extractor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    unet=UViT(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
        pretrained_path="models/uvit_v1.pdparams",
    ),
    vae=FrozenAutoencoderKL(
        pretrained_path="models/autoencoder_kl.pdparams"
    ),  # .from_pretrained('models/autoencoder_kl.pdparams'),
    scheduler=None,
)  # .from_pretrained("thu-ml/unidiffuser")
# pipe.remove_unused_weights()
pipe.vae = FrozenAutoencoderKL(pretrained_path="models/autoencoder_kl.pdparams")
image = "./unidiffuser-i.png"
image = pipe(image=image, generator=generator).images[0]
image.save("./unidiffuser-i2t2i.png")


import paddle

from ppdiffusers import UniDiffuserImageGenerationPipeline
from ppdiffusers.models import FrozenAutoencoderKL, UViT

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserImageGenerationPipeline(
    unet=UViT(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
        pretrained_path="models/uvit_v1.pdparams",
    ),
    vae=FrozenAutoencoderKL(
        pretrained_path="models/autoencoder_kl.pdparams"
    ),  # .from_pretrained('models/autoencoder_kl.pdparams'),
    scheduler=None,
)  # .from_pretrained("thu-ml/unidiffuser")
# pipe.remove_unused_weights()
image = pipe(generator=generator).images[0]
image.save("./unidiffuser-i.png")


import paddle

from ppdiffusers import CaptionDecoder, UniDiffuserTextGenerationPipeline, UViT

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserTextGenerationPipeline(
    unet=UViT(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
        pretrained_path="models/uvit_v1.pdparams",
    ),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64),
    scheduler=None,
)  # .from_pretrained("thu-ml/unidiffuser")
# pipe.remove_unused_weights()
pipe.caption_decoder = CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64)
text = pipe(generator=generator).texts[0]
with open("./unidiffuser-t.txt", "w") as f:
    print("{}\n".format(text), file=f)


import paddle

from ppdiffusers import (
    CaptionDecoder,
    FrozenAutoencoderKL,
    UniDiffuserJointGenerationPipeline,
    UViT,
)

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserJointGenerationPipeline(
    unet=UViT(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
        pretrained_path="models/uvit_v1.pdparams",
    ),
    vae=FrozenAutoencoderKL(
        pretrained_path="models/autoencoder_kl.pdparams"
    ),  # .from_pretrained('models/autoencoder_kl.pdparams'),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64),
    scheduler=None,
)  # .from_pretrained("thu-ml/unidiffuser")
# pipe.remove_unused_weights()
pipe.caption_decoder = CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64)
result = pipe(generator=generator)
image = result.images[0]
text = result.texts[0]
image.save("./unidiffuser-joint_i.png")
with open("./unidiffuser-joint_t.txt", "w") as f:
    print("{}\n".format(text), file=f)


import paddle

from ppdiffusers import (
    CaptionDecoder,
    FrozenCLIPEmbedder,
    UniDiffuserTextVariationPipeline,
    UViT,
)

generator = paddle.Generator().manual_seed(0)
# pipe = UniDiffuserTextVariationPipeline.from_pretrained("thu-ml/unidiffuser")
pipe = UniDiffuserTextVariationPipeline(
    clip_text_model=FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", max_length=77),
    unet=UViT(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
        pretrained_path="models/uvit_v1.pdparams",
    ),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64),
    scheduler=None,
)
pipe.clip_text_model = FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", max_length=77)
pipe.caption_decoder = CaptionDecoder(pretrained_path="models/caption_decoder.pdparams", hidden_dim=64)
prompt = "an astronaut riding on a horse on mars"
text = pipe(prompt=prompt, generator=generator).texts[0]
with open("./unidiffuser-t2i2t.txt", "w") as f:
    print("{}\n".format(text), file=f)
