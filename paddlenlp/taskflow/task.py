import abc
from abc import abstractmethod


class Task(metaclass=abc.ABCMeta):
    """ The meta classs of task in TaskFlow. The meta class has the five abstract function,
        the subclass need to inherit from the meta class.
    """

    def __init__(self, model, task, **kwargs):
        self.model = model
        self.task = task
        self.kwargs = kwargs

    @abstractmethod
    def _construct_model(self, model):
        """
       Construct the inference model for the predictor.
       """

    @abstractmethod
    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """

    @abstractmethod
    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
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
        inputs = self._preprocess(*args)
        outputs = self._run_model(inputs)
        return self._postprocess(outputs)
