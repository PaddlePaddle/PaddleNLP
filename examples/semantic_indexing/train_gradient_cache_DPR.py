import paddle
from paddle import nn
from paddle.nn import functional as F
from biencoder_base_model import BiEncoder, BiEncoderNllLoss
from NQdataset import BiEncoderPassage, BiEncoderSample, BiENcoderBatch, BertTensorizer, NQdataSetForDPR, DataUtil
from paddlenlp.transformers.bert.modeling import BertModel
import numpy as np
import os

global batch_size
global learning_rate
global weight_decay
global drop_out
global embedding_output_size
global data_path
global dataset
global chunk_numbers
global global_step
global save_steps
global save_direc

global_step = 0


data_path = "./dev.json"

batch_size = 8

drop_out = 0.1

embedding_output_size = 768

learning_rate = 1e-5

epoch = 5

def dataLoader_for_DPR(batch_size,source_data:list,epochs):
    index = np.arange(0,len(source_data))
    np.random.shuffle(index)
    batch_data = []
    for i in index:
        try:
            batch_data.append(source_data[i])

            if (len(batch_data) == batch_size):
                yield batch_data
                batch_data = []

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue


question_encoder = BertModel.from_pretrained("bert-base-uncased")
context_encoder = BertModel.from_pretrained("bert-base-uncased")

model = BiEncoder(question_encoder=question_encoder, context_encoder=context_encoder, dropout=drop_out,
                  output_emb_size=embedding_output_size)

# dataset = NQdataSetForDPR(data_path)
# data_loader = paddle.io.DataLoader(dataset,batch_size=batch_size,shuffle=True)

optimizer = paddle.optimizer.AdamW(
    learning_rate=learning_rate,
    parameters=model.parameters()
)

util = DataUtil()

LOSS = BiEncoderNllLoss()

batch_data = []

def train():

    for _ in range(epoch):

        index = np.arange(0, len(dataset))
        np.random.shuffle(index)

        for i in index:
            #dataLoader

            batch_data.append(dataset[i])
            if (len(batch_data) == batch_size):

                all_questions = []
                all_contexts = []
                all_positions = []
                all_CUDA_rnd_state = []

                chunked_x = [paddle.split(t, chunk_numbers, axis=0) for t in batch_data]

                batch_data = []

                sub_batchs = [list(s) for s in zip(*chunked_x)]


                for sub_batch in sub_batchs:

                    all_questions = []
                    all_contexts = []
                    all_positions = []
                    all_CUDA_rnd_state = []
                    #这里需要清空

                    sub_batch_input = util.create_biencoder_input(sub_batch)


                    with paddle.no_grad:

                        sub_CUDA_rnd_state = paddle.framework.random.get_cuda_rng_state(
                        )
                        # sub_global_rnd_state = paddle.framework.random.get_random_seed_generator(global_random_generator)

                        all_CUDA_rnd_state.append(sub_CUDA_rnd_state)
                        # all_global_rnd_state.append(sub_global_rnd_state)

                        sub_question_output = model.get_question_pooled_embedding(sub_batch_input.questions_ids,sub_batch_input.question_segments)

                        sub_context_ouput = model.get_context_pooled_embedding(sub_batch_input.context_ids,sub_batch_input.ctx_segments)


                        all_questions.append(sub_question_output)
                        all_contexts.append(sub_context_ouput)
                        all_positions.append(sub_batch_input.is_positive)

                model_questions = paddle.concat(all_questions, axis=0)

                model_questions.stop_gradient = False

                # Model_Repos = [r.detach() for r in model_reps]

                # Model_Repos.stop_gradient = False

                model_contexts = paddle.concat(all_contexts, axis=0)

                model_positions = paddle.concat(all_positions,axis=0)

                loss,_ = LOSS.calc(model_questions, model_contexts,model_positions)

                loss.backward()

                grads_for_questions = [question.grad for question in model_questions]
                grads_for_contexts = [context.grad for context in model_contexts]

                #all_grads = [repos.grad for repos in model_reps]
                #all_grads.append(model_reps.grad)

                for sub_batch_input, CUDA_state, grad_for_each_question,grad_for_each_context in zip(sub_batchs,
                                                   all_CUDA_rnd_state,
                                                   grads_for_questions,grads_for_contexts):

                    paddle.framework.random.set_cuda_rng_state(CUDA_state)
                    # paddle.framework.random.set_random_seed_generator(global_random_generator,global_rnd_state)

                    sub_question_output = model.get_question_pooled_embedding(sub_batch_input.questions_ids,
                                                                          sub_batch_input.question_segments)

                    sub_context_ouput = model.get_context_pooled_embedding(sub_batch_input.context_ids,
                                                                       sub_batch_input.ctx_segments)

                    finally_question_res_for_backward = paddle.dot(sub_question_output,grad_for_each_question)

                    finally_context_res_for_backward = paddle.dot(sub_context_ouput,grad_for_each_context)

                    finally_question_res_for_backward.backward()
                    finally_context_res_for_backward.backward()

                    # 反向传播

                    #梯度积累

                optimizer.step()
                optimizer.clear_grad()
                global_step = global_step + 1
                # 梯度更新

                if (global_step % save_steps == 0):
                    state = model.state_dict()
                    paddle.save()
                    save_dir = os.path.join(save_direc,
                                            "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    #tokenizer.save_pretrained(save_dir)
                    pass
                    #save models


if __name__ == '__main__':

    train()




