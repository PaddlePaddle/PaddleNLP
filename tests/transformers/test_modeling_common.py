import random
import copy

import tempfile
import numpy as np
import paddle

global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return paddle.reshape(paddle.to_tensor(data=values, dtype=paddle.int32),
                          shape=shape)


def random_attention_mask(shape, rng=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=rng)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return paddle.reshape(paddle.to_tensor(data=values, dtype=paddle.float32),
                          shape=shape)


class ModelTesterMixin:
    model_tester = None
    base_model_class = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_mismatched_shapes = True
    test_missing_keys = True
    is_encoder_decoder = False
    has_attentions = True
    model_split_percents = [0.5, 0.7, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__.endswith("ForMultipleChoice"):
            inputs_dict = {
                k: v.unsqueeze(1).expand(
                    shape=[-1, self.model_tester.num_choices, -1])
                if isinstance(v, paddle.Tensor) and v.ndim > 1 else v
                for k, v in inputs_dict.items()
            }
        return inputs_dict

    def test_save_load(self):
        config, input_ids, token_type_ids, input_mask = self.model_tester.prepare_config_and_inputs(
        )
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        for model_class in self.all_model_classes:
            if model_class == self.base_model_class:
                model = model_class(**config)
            else:
                model = model_class(self.base_model_class(**config))
            model.eval()
            with paddle.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class))

            out_2 = outputs[0].numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.eval()
                with paddle.no_grad():
                    after_outputs = model(
                        **self._prepare_for_class(inputs_dict, model_class))

                # Make sure we don't have nans
                out_1 = after_outputs[0].numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_base_model(self):
        pass
