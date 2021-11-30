from paddlenlp.utils.tools import static_params_to_dygraph
import paddle
from paddlenlp.transformers import ErnieModel, ErnieForPretraining, ErniePretrainingCriterion, ErnieTokenizer
import os

# device='gpu'
# paddle.enable_static()
# place = paddle.set_device(device)
# exe = paddle.static.Executor(place)
# startup_program = paddle.static.default_startup_program()
# exe.run(startup_program)

# main_program = paddle.static.default_main_program()
# model_name_or_path='./output/ernie-1.0-dp8-gb1024/model_200000'
# static_path = os.path.join(model_name_or_path, "static_vars")
# paddle.static.load(main_program, static_path, exe)
# model_path='output/model_200000/static_vars'
# static_model=paddle.static.load_program_state(model_path)
# print(static_model)
# model=static_params_to_dygraph()

import paddle
import paddle.static as static
import paddle.nn as nn

class SimpleModel(nn.Layer):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        self.l1=nn.Linear(784, 200)
        self.l2=nn.Linear(200, 10)

    def forward(self, x):
        x1=self.l1(x)
        x=self.l2(x1)
        return x

def static_mode():
    paddle.enable_static()
    model=SimpleModel()
    place = paddle.CPUPlace()
    exe = static.Executor(place)
    exe.run(static.default_startup_program())
    prog = static.default_main_program()
    static.save(prog, "./temp/model")
    program_state = static.load_program_state("./temp/model")
    ret_dict=static_params_to_dygraph(model,program_state)
    # print(ret_dict)

def dynamic_mode():
    model=SimpleModel()
    state_dict=paddle.load('./temp/model.pdparams')
    model.load_dict(state_dict)
    # print(model.parameters())

def load_ernie_model():

    model=ErnieModel.from_pretrained('ernie-1.0')
    # print(model.state_dict().keys())
    # state_dict=paddle.load('./output/ernie-1.0-dp8-gb1024/model_last/static_vars.pdparams')
    program_state = static.load_program_state("./output/ernie-1.0-dp8-gb1024/model_last/static_vars")
    # print(program_state)
    ret_dict=static_params_to_dygraph(model,program_state)

    # print(state_dict.keys())
    
    # print(model.state_dict())
    print('转换前的参数：')
    print(model.embeddings.word_embeddings.weight )
    model.load_dict(ret_dict)
    print('转换后的参数：')
    print(model.embeddings.word_embeddings.weight )
    model.save_pretrained("./ernie_checkpoint")
    # tokenizer.save_pretrained("./checkpoint')
    # print(ret_dict)
    


if __name__ == "__main__":
    # dynamic_mode()
    load_ernie_model()