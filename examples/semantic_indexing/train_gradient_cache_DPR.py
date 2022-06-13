import paddle
from paddle import nn
from paddle.nn import functional as F
from biencoder_base_model import BiEncoder, BiEncoderNllLoss
from NQdataset import BiEncoderPassage, BiEncoderSample, BiENcoderBatch, BertTensorizer, NQdataSetForDPR, DataUtil
from paddlenlp.transformers.bert.modeling import BertModel
import numpy as np
import os
import argparse
from paddle.optimizer.lr import LambdaDecay

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',required=True,type=int,default=None)
parser.add_argument('--learning_rate',required=True,type=int,default=None)
parser.add_argument('--save_dir',required=True,type=str,default=None)
parser.add_argument('--warmup_steps',required=True,type=int)
parser.add_argument('--epoches',required=True,type=int)
parser.add_argument('--max_grad_norm',required=True,type=int)
parser.add_argument('--train_data_path',required=True,type=str)
parser.add_argument('chunk_size',required=True,type=int)
args = parser.parse_args()

chunk_nums = args.batch_size // args.chunk_size
data_path = args.train_data_path
batch_size = args.batch_size
learning_rate = args.learning_rate
epoches = args.epoches

def dataLoader_for_DPR(batch_size, source_data: list, epochs):
    index = np.arange(0, len(source_data))
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

model = BiEncoder(question_encoder=question_encoder, context_encoder=context_encoder)

# dataset = NQdataSetForDPR(data_path)
# data_loader = paddle.io.DataLoader(dataset,batch_size=batch_size,shuffle=True)

def get_linear_scheduler(warmup_steps,training_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps))
        )
    return LambdaDecay(learning_rate=args.learning_rate,lr_lambda=lr_lambda(),last_epoch=-1,verbose=False)

training_steps = 58880  * args.epoches / args.batch_sizes
scheduler = get_linear_scheduler(args.warmup_steps,training_steps)
optimizer = paddle.optimizer.AdamW(
    learning_rate=scheduler,
    parameters=model.parameters()
)

util = DataUtil()

LOSS = BiEncoderNllLoss()

batch_data = []

data = NQdataSetForDPR(data_path)

dataset = data.new_data


def train():

    for epoch in range(epoches):

        index = np.arange(0, len(dataset))
        np.random.shuffle(index)

        batch_data = []

        for i in index:
            # dataLoader
            batch_data.append(dataset[i])
            if (len(batch_data) == batch_size):
                all_questions = []
                all_contexts = []
                # all_positions = []
                all_CUDA_rnd_state = []

                all_batch_input = util.create_biencoder_input(batch_data)

                all_positions = all_batch_input.is_positive

                all_inputs_questions_id = all_batch_input.questions_ids
                all_inputs_questions_segment = all_batch_input.question_segments

                all_inputs_contexts_id = all_batch_input.context_ids
                all_inputs_contexts_segment = all_batch_input.ctx_segments

                sub_q_ids = paddle.split(all_inputs_questions_id, chunk_nums, axis=0)
                sub_c_ids = paddle.split(all_inputs_contexts_id, chunk_nums, axis=0)
                sub_q_segments = paddle.split(all_inputs_questions_segment, chunk_nums, axis=0)
                sub_c_segments = paddle.split(all_inputs_contexts_segment, chunk_nums, axis=0)

                # chunked_x = [paddle.split(t, chunk_numbers, axis=0) for t in batch_data]

                # sub_batchs = [list(s) for s in zip(*chunked_x)]

                all_questions = []
                all_contexts = []
                all_CUDA_rnd_state = []

                for sub_q_id, sub_c_id, sub_q_segment, sub_c_segment in zip(sub_q_ids, sub_c_ids, sub_q_segments,
                                                                            sub_c_segments):
                    # sub_batch_input = util.create_biencoder_input(sub_batch)
                    with paddle.no_grad():
                        sub_CUDA_rnd_state = paddle.framework.random.get_cuda_rng_state(
                        )
                        # sub_global_rnd_state = paddle.framework.random.get_random_seed_generator(global_random_generator)

                        all_CUDA_rnd_state.append(sub_CUDA_rnd_state)
                        # all_global_rnd_state.append(sub_global_rnd_state)

                        sub_question_output = model.get_question_pooled_embedding(sub_q_id, sub_q_segment)

                        sub_context_ouput = model.get_context_pooled_embedding(sub_c_id, sub_c_segment)

                        all_questions.append(sub_question_output)
                        all_contexts.append(sub_context_ouput)
                        # all_positions.append(sub_batch_input.is_positive)

                model_questions = paddle.concat(all_questions, axis=0)
                all_questions = []

                model_questions = model_questions.detach()

                model_questions.stop_gradient = False

                # Model_Repos = [r.detach() for r in model_reps]

                # Model_Repos.stop_gradient = False

                model_contexts = paddle.concat(all_contexts, axis=0)

                model_contexts = model_contexts.detach()

                model_contexts.stop_gradient = False

                all_contexts = []

                model_positions = all_positions

                loss, _ = LOSS.calc(model_questions, model_contexts, model_positions)

                print("损失是：")

                print(loss)

                loss.backward()

                """grads_for_questions = [question for question in model_questions.grad]
                grads_for_contexts = [context for context in model_contexts.grad]"""

                grads_for_questions = paddle.split(model_questions.grad, chunk_nums, axis=0)
                grads_for_contexts = paddle.split(model_contexts.grad, chunk_nums, axis=0)

                # all_grads = [repos.grad for repos in model_reps]
                # all_grads.append(model_reps.grad)

                for sub_q_id, sub_c_id, sub_q_segment, sub_c_segment, CUDA_state, grad_for_each_question, grad_for_each_context in zip(
                        sub_q_ids,
                        sub_c_ids,
                        sub_q_segments,
                        sub_c_segments,
                        all_CUDA_rnd_state,
                        grads_for_questions,
                        grads_for_contexts
                ):
                    paddle.framework.random.set_cuda_rng_state(CUDA_state)

                    # paddle.framework.random.set_random_seed_generator(global_random_generator,global_rnd_state)

                    sub_question_output = model.get_question_pooled_embedding(sub_q_id,
                                                                              sub_q_segment)

                    sub_context_ouput = model.get_context_pooled_embedding(sub_c_id,
                                                                           sub_c_segment)

                    finally_question_res_for_backward = paddle.dot(sub_question_output, grad_for_each_question)
                    finally_context_res_for_backward = paddle.dot(sub_context_ouput, grad_for_each_context)

                    finally_question_res_for_backward = finally_question_res_for_backward * (1/8.)
                    finally_context_res_for_backward = finally_context_res_for_backward * (1/8.)

                    finally_question_res_for_backward.backward(retain_graph=True)
                    finally_context_res_for_backward.backward(retain_graph=True)

                    """print(finally_question_res_for_backward)
                    print(finally_context_res_for_backward)"""



                paddle.nn.ClipGradByGlobalNorm(clip_norm=args.max_grad_norm, group_name=model.parameters())
                optimizer.step()
                scheduler.step()
                optimizer.clear_grad()
                all_CUDA_rnd_state = []

                batch_data = []




        EPOCH = str(epoch)
        model.question_encoder.save_pretrained('./question_model'/{EPOCH})
        model.context_encoder.save_pretrained('./context_model'/{EPOCH})





"""if (global_step % save_steps == 0):
                    state = model.state_dict()
                    paddle.save()
                    save_dir = os.path.join(save_direc,
                                            "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)"""
# tokenizer.save_pretrained(save_dir)
# pass
# save models

if __name__ == '__main__':
    train()




