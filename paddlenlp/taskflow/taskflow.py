import warnings
from .text2knowledge import Text2KnowledgeTask
from .sentiment_analysis import SentaTask

warnings.simplefilter(action='ignore', category=Warning, lineno=0, append=False)

TASKS = {
    "text2knowledge": {
        "models": {
            "wordtag": {
                "model_class": Text2KnowledgeTask,
                "max_seq_len": 128,
            }
        },
        "default": {
            "model": "wordtag"
        }
    },
    'sentiment_analysis': {
        "models": {
            "senta": {
                "model_class": SentaTask
            },
        },
        "default": {
            "model": "senta"
        }
    }
}


class TaskFlow(object):
    """
    The TaskFlow is the end2end inferface that could convert the raw text to model result, and decode the model result to task result. The main functions as follows:
        1) Convert the raw text to task result.
        2) Convert the model to the inference model.
        3) Offer the usesage and help message.
    Args:
        task (str): The task name for the TaskFlow, and get the task class from the name.
        model (str, optional): The model name in the task, if set None, will use the default model.  
        device_id (int, optional): The device id for the gpu, xpu and other devices, the defalut value is 0.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 

    """

    def __init__(self, task, model=None, device_id=0, **kwargs):
        assert task in TASKS, "The task name:{} is not in TaskFlow list, please check your task name.".format(
            task)
        self.task = task
        if model is not None:
            assert model in set(TASKS[task]['models'].keys(
            )), "The model name:{} is not in task:[{}]".format(model)
        else:
            model = TASKS[task]['default']['model']
        self.model = model
        config_kwargs = TASKS[self.task]['models'][self.model]
        kwargs.update(config_kwargs)
        self.kwargs = kwargs
        model_class = TASKS[self.task]['models'][self.model]['model_class']
        self.model_instance = model_class(
            model=self.model, task=self.task, **self.kwargs)

    def __call__(self, *inputs):
        results = self.model_instance(inputs)
        return results

    def help(self):
        pass
