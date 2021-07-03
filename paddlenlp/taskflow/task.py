import abc
from abc import ABC, abstractmethod


class Task(metaclass=abc.ABCMeta):
    def __init__(self, model, task, **kwargs):
        self.model = model
        self.task = task
        self.kwargs = kwargs

    @abstractmethod
    def _construct_model_tokenizer(self, model, **kwargs):
        """
       Construct the tokenizer and model for the predictor.
       """

    @abstractmethod
    def _text_tokenize(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """

    @abstractmethod
    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """

    @abstractmethod
    def _postprocess(self, inputs):
        """
       The model output is allways the logits and pros, this function will convert the model output to raw text.
       """

    def __call__(self, *args):
        inputs = self._text_tokenize(*args)
        outputs = self._run_model(inputs)
        return self._postprocess(outputs)
