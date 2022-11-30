# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is rewritten by Paddle based on Jina-ai/discoart.
https://github.com/jina-ai/discoart/blob/main/discoart/runner.py
"""
import paddle
import gc
import random
import numpy as np
import paddle
import paddle.vision.transforms as T

from PIL import Image
from pathlib import Path
from paddle.utils import try_import
from .losses import range_loss, spherical_dist_loss, tv_loss
from .make_cutouts import MakeCutoutsDango
from .sec_diff import alpha_sigma_to_t
from .transforms import Normalize
from .perlin_noises import create_perlin_noise, regen_perlin
import random
from ..image_utils import load_image

__all__ = ["DiscoDiffusionMixin"]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)


class DiscoDiffusionMixin:
    def disco_diffusion_generate(
        self,
        target_text_embeds,
        init_image=None,
        output_dir="outputs/",
        width_height=[1280, 768],
        skip_steps=0,
        cut_ic_pow=1,
        init_scale=1000,
        clip_guidance_scale=5000,
        tv_scale=0,
        range_scale=0,
        sat_scale=0,
        cutn_batches=4,
        perlin_init=False,
        perlin_mode="mixed",
        seed=None,
        eta=0.8,
        clamp_grad=True,
        clamp_max=0.05,
        cut_overview="[12]*400+[4]*600",
        cut_innercut="[4]*400+[12]*600",
        cut_icgray_p="[0.2]*400+[0]*600",
        save_rate=10,
        n_batches=1,
        batch_name="",
        use_secondary_model=True,
        randomize_class=True,
        clip_denoised=False,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    ):
        r"""
        The DiffusionMixin diffusion_generate method.

        Args:
            init_image (Path, optional):
                Recall that in the image sequence above, the first image shown is just noise.  If an init_image
                is provided, diffusion will replace the noise with the init_image as its starting state.  To use
                an init_image, upload the image to the Colab instance or your Google Drive, and enter the full
                image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total
                steps to retain the character of the init. See skip_steps above for further discussion.
                Default to `None`.
            output_dir (Path, optional):
                Output directory.
                Default to `disco_diffusion_clip_vitb32_out`.
            width_height (List[int, int], optional):
                Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge
                length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.
                If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your
                image to make it so.
                Default to `[1280, 768]`.
            skip_steps (int, optional):
                Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively
                gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high,
                so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the
                amount an image changes per step) declines, and image coherence from one step to the next increases.
                The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be
                skipped without affecting the final image. You can experiment with this as a way to cut render times.
                If you skip too many steps, however, the remaining noise may not be high enough to generate new content,
                and thus may not have time left to finish an image satisfactorily.Also, depending on your other settings,
                you may need to skip steps to prevent CLIP from overshooting your goal, resulting in blown out colors
                (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that
                the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate
                other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain
                the shapes in the original init image. However, if you're using an init_image, you can also adjust
                skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by"
                the init_image which will retain the colors and rough layout and shapes but look quite different.
                With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
                Default to `0`.
            steps:
                When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration)
                involves the AI looking at subsets of the image called 'cuts' and calculating the 'direction' the image
                should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser,
                and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image,
                and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps
                comes at the expense of longer render times.  Also, while increasing steps should generally increase image
                quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate
                images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is
                directly related to the number of steps, and many other parameters have a major impact on image quality, without
                costing additional time.
            cut_ic_pow (int, optional):
                This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and
                therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small
                inner cuts, you may lose overall image coherency and/or it may cause an undesirable 'mosaic' effect.
                Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping
                with some details.
                Default to `1`.
            init_scale (int, optional):
                This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the
                clip_guidance_scale (CGS) above.  Too much init scale, and the image won't change much during diffusion.
                Too much CGS and the init image will be lost.
                Default to `1000`.
            clip_guidance_scale (int, optional):
                CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move
                toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot
                the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust
                CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your
                total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the
                image, you'd want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale,
                steps and skip_steps are the most important contributors to image quality, so learn them well.
                Default to `5000`.
            tv_scale (int, optional):
                Total variance denoising. Optional, set to zero to turn off. Controls smoothness of final output. If used,
                tv_scale will try to smooth out your final image to reduce overall noise. If your image is too 'crunchy',
                increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.
                See https://en.wikipedia.org/wiki/Total_variation_denoising
                Default to `0`.
            range_scale (int, optional):
                Optional, set to zero to turn off.  Used for adjustment of color contrast. Lower range_scale will increase
                contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images.
                Higher range_scale will reduce contrast, for more muted images.
                Default to `0`.
            sat_scale (int, optional):
                Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation.
                If your image is too saturated, increase sat_scale to reduce the saturation.
                Default to `0`.
            cutn_batches (int, optional):
                Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt
                to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has
                more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however,
                and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase
                cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per
                timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if
                cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches
                of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts,
                but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to
                render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will
                increase render times, however, as the work is being done sequentially.  DD's default cut schedule is a good place
                to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.
                Default to `4`.
            perlin_init (bool, optional):
                Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.
                If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very
                interesting characteristics, distinct from random noise, so it's worth experimenting with this for your projects.
                Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an
                init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the
                starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image
                you may have specified. Further, because the 2D, 3D and video animation systems all rely on the init_image system,
                if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video
                input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and
                animation modes together do make a very colorful rainbow effect, which can be used creatively.
                Default to `False`.
            perlin_mode (str, optional):
                sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment
                to see what these do in your projects.
                Default to `mixed`.
            seed (int, optional):
                Deep in the diffusion code, there is a random number seed which is used as the basis for determining the initial
                state of the diffusion.  By default, this is random, but you can also specify your own seed. This is useful if you like a
                particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be
                reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used
                repeatedly, the resulting images will be quite similar but not identical.
                Default to `None`.
            eta (float, optional):
                Eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep.
                0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you
                unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0,
                then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around
                250 and up. eta has a subtle, unpredictable effect on image, so you'll need to experiment to see how this affects your projects.
                Default to `0.8`.
            clamp_grad (bool, optional):
                As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results. Try your images with and without
                clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and
                should be reduced.
                Default to `True`.
            clamp_max (float, optional):
                Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting
                higher values (0.15-0.3) can provide interesting contrast and vibrancy.
                Default to `0.05`.
            cut_overview (str, optional):
                The schedule of overview cuts.
                Default to `'[12]*400+[4]*600'`.
            cut_innercut (str, optional):
                The schedule of inner cuts.
                Default to `'[4]*400+[12]*600'`.
            cut_icgray_p (str, optional):
                This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts
                themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall
                image coherency and/or it may cause an undesirable 'mosaic' effect.   Low cut_ic_pow values will allow the inner cuts to be
                larger, helping image coherency while still helping with some details.
                Default to `'[0.2]*400+[0]*600'`.
            save_rate (int, optional):
                During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set
                to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way
                to get an early peek at where your image is heading. If you don't like the progression, just interrupt execution, change some
                settings, and re-run.  If you are planning a long, unmonitored batch, it's better to set display_rate equal to steps, because
                displaying interim images does slow Colab down slightly.
                Default to `10`.
            n_batches (int, optional):
                This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details)
                DD will ignore n_batches and create a single set of animated frames based on the animation settings.
                Default to `1`.
            batch_name (str, optional):
                The name of the batch, the batch id will be named as "progress-[batch_name]-seed-[range(n_batches)]-[save_rate]". To avoid your
                artworks be overridden by other users, please use a unique name.
                Default to `''`.
            use_secondary_model (bool, optional):
                Whether or not use secondary model.
                Default to `True`.
            randomize_class (bool, optional):
                Random class.
                Default to `True`.
            clip_denoised (bool, optional):
                Clip denoised.
                Default to `False`.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        batch_size = 1
        normalize = Normalize(
            mean=image_mean,
            std=image_std,
        )
        side_x = (width_height[0] // 64) * 64
        side_y = (width_height[1] // 64) * 64
        cut_overview = eval(cut_overview)
        cut_innercut = eval(cut_innercut)
        cut_icgray_p = eval(cut_icgray_p)

        seed = seed or random.randint(0, 2**32)
        set_seed(seed)

        init = None
        if init_image:
            d = load_image(init_image)
            init = T.to_tensor(d).unsqueeze(0) * 2 - 1

        if perlin_init:
            if perlin_mode == "color":
                init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_y, side_x)
                init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False, side_y, side_x)
            elif perlin_mode == "gray":
                init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True, side_y, side_x)
                init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x)
            else:
                init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_y, side_x)
                init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x)
            init = T.to_tensor(init).add(T.to_tensor(init2)).divide(paddle.to_tensor(2.0)).unsqueeze(0) * 2 - 1
            del init2

        if init is not None and init_scale:
            lpips = try_import("paddle_lpips")
            lpips_model = lpips.LPIPS(net="vgg")
            lpips_model.eval()
            for parameter in lpips_model.parameters():
                parameter.stop_gradient = True

        cur_t = None

        def cond_fn(x, t, y=None):
            x_is_NaN = False
            n = x.shape[0]
            x = paddle.to_tensor(x.detach(), dtype="float32")
            x.stop_gradient = False
            if use_secondary_model:
                alpha = paddle.to_tensor(self.diffusion.sqrt_alphas_cumprod[cur_t], dtype="float32")
                sigma = paddle.to_tensor(self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t], dtype="float32")
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                cosine_t = paddle.tile(paddle.to_tensor(cosine_t.detach().cpu().numpy()), [n])
                cosine_t.stop_gradient = False
                out = self.secondary_model(x, cosine_t).pred
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in_d = out * fac + x * (1 - fac)
                x_in = x_in_d.detach()
                x_in.stop_gradient = False
                x_in_grad = paddle.zeros_like(x_in, dtype="float32")
            else:
                t = paddle.ones([n], dtype="int64") * cur_t
                out = self.diffusion.p_mean_variance(self.unet_model, x, t, clip_denoised=False, model_kwargs={"y": y})
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in_d = out["pred_xstart"].astype("float32") * fac + x * (1 - fac)
                x_in = x_in_d.detach()
                x_in.stop_gradient = False
                x_in_grad = paddle.zeros_like(x_in, dtype="float32")

            for _ in range(cutn_batches):
                t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                try:
                    input_resolution = self.vision_model.input_resolution
                except:
                    input_resolution = 224

                cuts = MakeCutoutsDango(
                    input_resolution,
                    Overview=cut_overview[1000 - t_int],
                    InnerCrop=cut_innercut[1000 - t_int],
                    IC_Size_Pow=cut_ic_pow,
                    IC_Grey_P=cut_icgray_p[1000 - t_int],
                )
                clip_in = normalize(cuts(x_in.add(paddle.to_tensor(1.0)).divide(paddle.to_tensor(2.0))))
                image_embeds = self.get_image_features(clip_in)

                dists = spherical_dist_loss(
                    image_embeds.unsqueeze(1),
                    target_text_embeds.unsqueeze(0),
                )

                dists = dists.reshape(
                    [
                        cut_overview[1000 - t_int] + cut_innercut[1000 - t_int],
                        n,
                        -1,
                    ]
                )
                losses = dists.sum(2).mean(0)
                x_in_grad += paddle.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(x_in)
            sat_losses = paddle.abs(x_in - x_in.clip(min=-1, max=1)).mean()
            loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += paddle.grad(loss, x_in)[0]
            if not paddle.isnan(x_in_grad).any():
                grad = -paddle.grad(x_in_d, x, x_in_grad)[0]
            else:
                x_is_NaN = True
                grad = paddle.zeros_like(x)
            if clamp_grad and not x_is_NaN:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clip(max=clamp_max) / magnitude
            return grad

        # we use ddim sample
        sample_fn = self.diffusion.ddim_sample_loop_progressive

        da_batches = []

        # process output file name
        output_filename_list = ["progress"]
        if batch_name != "":
            output_filename_list.append(batch_name)
        if seed is not None:
            output_filename_list.append(str(seed))
        output_filename_prefix = "-".join(output_filename_list)

        for _nb in range(n_batches):
            gc.collect()
            paddle.device.cuda.empty_cache()
            cur_t = self.diffusion.num_timesteps - skip_steps - 1

            if perlin_init:
                init = regen_perlin(perlin_mode, side_y, side_x, batch_size)

            samples = sample_fn(
                self.unet_model,
                (batch_size, 3, side_y, side_x),
                clip_denoised=clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=randomize_class,
                eta=eta,
            )

            for j, sample in enumerate(samples):
                cur_t -= 1
                if j % save_rate == 0 or cur_t == -1:
                    for b, image in enumerate(sample["pred_xstart"]):
                        image = (((image + 1) / 2).clip(0, 1).squeeze().transpose([1, 2, 0]).numpy() * 255).astype(
                            "uint8"
                        )
                        image = Image.fromarray(image)
                        image.save(output_dir / f"{output_filename_prefix}-{_nb}-{j}.png")
                        if cur_t == -1:
                            da_batches.append(image)

        return da_batches
