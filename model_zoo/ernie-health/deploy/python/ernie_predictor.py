# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import os
import numpy as np
import paddle
from multiprocessing import cpu_count
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers import normalize_chars, tokenize_special_chars

import time


class InferBackend(object):
    def __init__(self,
                 model_path,
                 batch_size=32,
                 device='cpu',
                 use_fp16=False,
                 use_quantize=False,
                 set_dynamic_shape=False,
                 shape_info_file='shape_info.txt',
                 num_threads=10):

        if device == 'gpu':
            int8_model = self.paddle_quantize_model(model_path)
            if int8_model and use_fp16:
                print(
                    '>>> [InferBackend] load a paddle quantize model, use_fp16 has been closed...'
                )
                use_fp16 = False

            if int8_model or use_fp16:
                from paddle import inference
                self.predictor_type = 'inference'
                config = inference.Config(model_path + '.pdmodel',
                                          model_path + '.pdiparams')
                config.enable_use_gpu(100, 0)
                paddle.set_device("gpu")

                if use_fp16:
                    assert device == 'gpu', 'When use_fp16, please set device to gpu and install requirements_gpu.txt.'
                    print('>>> [InferBackend] FP16 inference ...')
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        precision_mode=inference.PrecisionType.Half,
                        max_batch_size=batch_size,
                        min_subgraph_size=5,
                        use_static=True,
                        use_calib_mode=False)

                if int8_model:
                    print('>>> [InferBackend] INT8 inference ...')
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        precision_mode=inference.PrecisionType.Int8,
                        max_batch_size=batch_size,
                        min_subgraph_size=5,
                        use_static=False,
                        use_calib_mode=False)

                if set_dynamic_shape:
                    config.collect_shape_range_info(shape_info_file)
                else:
                    config.enable_tuned_tensorrt_dynamic_shape(shape_info_file,
                                                               True)

                self.predictor = inference.create_predictor(config)
                self.input_names = [
                    name for name in self.predictor.get_input_names()
                ]
                self.input_handles = [
                    self.predictor.get_input_handle(name)
                    for name in self.predictor.get_input_names()
                ]
                self.output_handles = [
                    self.predictor.get_output_handle(name)
                    for name in self.predictor.get_output_names()
                ]
            else:
                import paddle2onnx
                import onnxruntime as ort
                import copy
                self.predictor_type = 'onnxruntime'
                onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                    model_file=model_path + '.pdmodel',
                    params_file=model_path + '.pdiparams',
                    opset_version=13,
                    enable_onnx_checker=True)
                providers = ['CUDAExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = num_threads
                sess_options.inter_op_num_threads = num_threads
                self.predictor = ort.InferenceSession(
                    onnx_model, sess_options=sess_options, providers=providers)
                self.input_handles = [
                    self.predictor.get_inputs()[0].name,
                    self.predictor.get_inputs()[1].name,
                    self.predictor.get_inputs()[2].name
                ]
                self.output_handles = []
        else:
            import paddle2onnx
            import onnxruntime as ort
            import copy
            self.predictor_type = 'onnxruntime'
            dynamic_quantize_model = paddle2onnx.command.c_paddle_to_onnx(
                model_file=model_path + '.pdmodel',
                params_file=model_path + '.pdiparams',
                opset_version=13,
                enable_onnx_checker=True)
            providers = ['CPUExecutionProvider']
            if use_quantize:
                float_onnx_file = "model.onnx"
                with open(float_onnx_file, "wb") as f:
                    f.write(dynamic_quantize_model)
                dynamic_quantize_model = "dynamic_quantize_model.onnx"
                self.dynamic_quantize(float_onnx_file, dynamic_quantize_model)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
            self.predictor = ort.InferenceSession(
                dynamic_quantize_model,
                sess_options=sess_options,
                providers=providers)
            self.input_handles = [
                self.predictor.get_inputs()[0].name,
                self.predictor.get_inputs()[1].name,
                self.predictor.get_inputs()[2].name
            ]
            self.output_handles = []

    def dynamic_quantize(self, input_float_model, dynamic_quantized_model):
        from onnxruntime.quantization import QuantizationMode, quantize_dynamic
        quantize_dynamic(input_float_model, dynamic_quantized_model)

    def paddle_quantize_model(self, model_path):
        model = paddle.jit.load(model_path)
        program = model.program()
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type.count("quantize"):
                    return True
        return False

    def infer(self, input_dict: dict):
        if self.predictor_type == "inference":
            for idx, input_name in enumerate(self.input_names):
                self.input_handles[idx].copy_from_cpu(input_dict[input_name])
            self.predictor.run()
            output = [
                output_handle.copy_to_cpu()
                for output_handle in self.output_handles
            ]
            return output

        result = self.predictor.run(None, input_dict)
        return result


LABEL_LIST = {
    'qic': [
        '病情诊断', '治疗方案', '病因分析', '指标解读', '就医建议', '疾病表述', '后果表述', '注意事项', '功效作用',
        '医疗费用', '其他'
    ],
    'qtr': ['完全不匹配', '很少匹配，有一些参考价值', '部分匹配', '完全匹配'],
    'qqr': [
        'B为A的语义父集，B指代范围大于A； 或者A与B语义毫无关联。', 'B为A的语义子集，B指代范围小于A。',
        '表示A与B等价，表述完全一致。'
    ],
    'ctc': [
        '成瘾行为', '居住情况', '年龄', '酒精使用', '过敏耐受', '睡眠', '献血', '能力', '依存性', '知情同意',
        '数据可及性', '设备', '诊断', '饮食', '残疾群体', '疾病', '教育情况', '病例来源', '参与其它试验',
        '伦理审查', '种族', '锻炼', '性别', '健康群体', '实验室检查', '预期寿命', '读写能力', '含有多类别的语句',
        '肿瘤进展', '疾病分期', '护理', '口腔相关', '器官组织状态', '药物', '怀孕相关', '受体状态', '研究者决定',
        '风险评估', '性取向', '体征(医生检测）', ' 吸烟状况', '特殊病人特征', '症状(患者感受)', '治疗或手术'
    ],
    'sts': ['语义不同', '语义相同'],
    'cdn': ['否', '是'],
    'cmeee': [[
        'B-bod', 'I-bod', 'E-bod', 'S-bod', 'B-dis', 'I-dis', 'E-dis', 'S-dis',
        'B-pro', 'I-pro', 'E-pro', 'S-pro', 'B-dru', 'I-dru', 'E-dru', 'S-dru',
        'B-ite', 'I-ite', 'E-ite', 'S-ite', 'B-mic', 'I-mic', 'E-mic', 'S-mic',
        'B-equ', 'I-equ', 'E-equ', 'S-equ', 'B-dep', 'I-dep', 'E-dep', 'S-dep',
        'O'
    ], ['B-sym', 'I-sym', 'E-sym', 'S-sym', 'O']],
    'cmeie': [
        '预防', '阶段', '就诊科室', '辅助治疗', '化疗', '放射治疗', '手术治疗', '实验室检查', '影像学检查',
        '辅助检查', '组织学检查', '内窥镜检查', '筛查', '多发群体', '发病率', '发病年龄', '多发地区', '发病性别倾向',
        '死亡率', '多发季节', '传播途径', '并发症', '病理分型', '相关（导致）', '鉴别诊断', '相关（转化）',
        '相关（症状）', '临床表现', '治疗后症状', '侵及周围组织转移的症状', '病因', '高危因素', '风险评估因素', '病史',
        '遗传因素', '发病机制', '病理生理', '药物治疗', '发病部位', '转移部位', '外侵部位', '预后状况', '预后生存率',
        '同义词'
    ]
}

DESCRIPTIONS = {
    'qic': '医疗搜索检索词意图分类',
    'qtr': '医疗搜索查询词-页面标题相关性',
    'qqr': '医疗搜索查询词-查询词相关性',
    'ctc': '临床试验筛选标准短文本分类',
    'sts': '平安医疗科技疾病问答迁移学习',
    'cdn': '临床术语标准化',
    'cmeee': '中文医学命名实体识别',
    'cmeie': '中文医学文本实体关系抽取'
}


class ErnieHealthPredictor(object):
    def __init__(self, args):
        if not isinstance(args.device, six.string_types):
            print(
                ">>> [InferBackend] The type of device must be string, but the type you set is: ",
                type(device))
            exit(0)
        args.device = args.device.lower()
        if args.device not in ['cpu', 'gpu']:
            print(
                ">>> [InferBackend] The device must be cpu or gpu, but your device is set to:",
                type(args.device))
            exit(0)

        self.task_name = args.task_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_faster=True)

        if args.task_name in ['qic', 'qtr', 'qqr', 'ctc', 'sts', 'cdn']:
            self.preprocess = self.cblue_cls_preprocess
            self.postprocess = self.cblue_cls_postprocess
            self.printer = self.cblue_cls_print_ret
        elif args.task_name == 'cmeee':
            self.preprocess = self.cblue_token_preprocess
            self.postprocess = self.cblue_ner_postprocess
            self.printer = self.cblue_ner_print_ret
        elif args.task_name == 'cmeie':
            self.preprocess = self.cblue_token_preprocess
            self.postprocess = self.cblue_spo_postprocess
            self.printer = self.cblue_spo_print_ret
        else:
            print("[ErnieHealthPredictor]: task_name only support CBLUE 1.0,",
                  "including qic, qtr, qqr, ctc, sts, cdn, cmeee, cmeie.")
            exit(0)

        self.max_seq_length = args.max_seq_length

        if args.device == 'cpu':
            args.use_fp16 = False
            args.set_dynamic_shape = False
            args.batch_size = 32
            args.shape_info_file = None
        if args.device == 'gpu':
            args.num_threads = cpu_count()
            args.use_quantize = False
        self.inference_backend = InferBackend(
            args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            use_fp16=args.use_fp16,
            use_quantize=args.use_quantize,
            set_dynamic_shape=args.set_dynamic_shape,
            shape_info_file=args.shape_info_file,
            num_threads=args.num_threads)
        if args.set_dynamic_shape:
            # If set_dynamic_shape is turned on, all required dynamic shapes will be automatically set according to the batch_size and max_seq_length.
            self.set_dynamic_shape(args.max_seq_length, args.batch_size)
            exit(0)

    def cblue_cls_print_ret(self, infer_result, input_datas):
        label_list = LABEL_LIST[self.task_name]
        label = infer_result["label"].squeeze()
        confidence = infer_result["confidence"].squeeze()
        for i, ret in enumerate(infer_result):
            print("Input data:", input_datas[i])
            print(DESCRIPTIONS[self.task_name] + ":")
            print("Label:", label_list[label[i]], "  Confidence:",
                  confidence[i])
            print("-----------------------------")

    def cblue_ner_print_ret(self, infer_result, input_datas):
        for i, ret in enumerate(infer_result):
            print("Input data:", input_datas[i])
            print("Detected entities:")
            for item in ret:
                print("Type:", item["type"],
                      "Position: (%d, %d)" % (item["start_id"], item["end_id"]),
                      "Entity: ", item["entity"])
            print("-----------------------------")

    def cblue_spo_print_ret(self, infer_result, input_datas):
        labels = LABEL_LIST['cmeie']
        ents, spos = infer_result['entity'], infer_result['spo']
        for i, (ent, rel) in enumerate(zip(ents, spos)):
            text = input_datas[i]
            print("Input data:")
            print(text)
            print("Detected entities and relations:")
            for sid, eid in ent:
                print('    Entity: %s, Position: (%d, %d)' %
                      (input_datas[i][sid:eid], sid, eid))
            for s, p, o in rel:
                print('    SPO:', text[s[0]:s[1]], labels[p], text[o[0]:o[1]])
            print("-----------------------------")

    def cblue_cls_preprocess(self, input_data: list):
        norm_text = lambda x: tokenize_special_chars(normalize_chars(x))
        if isinstance(input_data[0], list):
            data = [[norm_text(x) for x in sample] for sample in input_data]
        else:
            data = [norm_text(x) for x in input_data]

        data = self.tokenizer(
            data,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_position_ids=True)

        input_ids = np.array(data["input_ids"], dtype="int64")
        token_type_ids = np.array(data["token_type_ids"], dtype="int64")
        position_ids = np.array(data['position_ids'], dtype="int64")

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids
        }

    def cblue_token_preprocess(self, input_data: list):
        data = self.tokenizer(
            input_data,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_position_ids=True)

        input_ids = np.array(data["input_ids"], dtype="int64")
        token_type_ids = np.array(data["token_type_ids"], dtype="int64")
        position_ids = np.array(data['position_ids'], dtype="int64")

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": np.ones_like(
                input_ids, dtype="float32")
        }

    def cblue_cls_postprocess(self, infer_data, input_data):
        logits = np.array(infer_data[0])
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {
            "label": probs.argmax(axis=-1),
            "confidence": probs.max(axis=-1)
        }
        return out_dict

    def cblue_ner_postprocess(self, infer_data, input_data):
        en_to_cn = {
            'bod': '身体',
            'mic': '微生物类',
            'dis': '疾病',
            'sym': '临床表现',
            'pro': '医疗程序',
            'equ': '医疗设备',
            'dru': '药物',
            'dep': '科室',
            'ite': '医学检验项目'
        }

        def extract_chunk(text, tokens):
            chunks = set()
            start_idx, cur_idx = 0, 0
            while cur_idx < len(tokens):
                if tokens[cur_idx][0] == 'B':
                    start_idx = cur_idx
                    cur_idx += 1
                    while cur_idx < len(tokens) and tokens[cur_idx][0] == 'I':
                        if tokens[cur_idx][2:] == tokens[start_idx][2:]:
                            cur_idx += 1
                        else:
                            break
                    if cur_idx < len(tokens) and tokens[cur_idx][0] == 'E':
                        if tokens[cur_idx][2:] == tokens[start_idx][2:]:
                            chunks.add((tokens[cur_idx][2:], start_idx - 1,
                                        cur_idx, text[start_idx - 1:cur_idx]))
                            cur_idx += 1
                elif tokens[cur_idx][0] == 'S':
                    chunks.add((tokens[cur_idx][2:], cur_idx - 1, cur_idx,
                                text[cur_idx - 1]))
                    cur_idx += 1
                else:
                    cur_idx += 1

            return list(chunks)

        tokens_oth = np.argmax(infer_data[0], axis=-1)
        tokens_sym = np.argmax(infer_data[1], axis=-1)
        entity = []
        for oth_ids, sym_ids, text in zip(tokens_oth, tokens_sym, input_data):
            token_oth = [LABEL_LIST['cmeee'][0][x] for x in oth_ids]
            token_sym = [LABEL_LIST['cmeee'][1][x] for x in sym_ids]
            sub_entity = []
            chunks = extract_chunk(text, token_oth) + extract_chunk(text,
                                                                    token_sym)
            for etype, sid, eid, name in chunks:
                sub_entity.append({
                    'type': en_to_cn[etype],
                    'start_id': sid,
                    'end_id': eid,
                    'entity': name
                })
            entity.append(sub_entity)
        return entity

    def cblue_spo_postprocess(self, infer_data, input_data):
        ent_logits = np.array(infer_data[0])
        spo_logits = np.array(infer_data[1])
        ent_pred_list = []
        ent_idxs_list = []
        for batch_id, ent_pred in enumerate(ent_logits):
            seq_len = len(ent_pred)
            start = np.where(ent_pred[:, 0] > 0.5)[0]
            end = np.where(ent_pred[:, 1] > 0.5)[0]
            ent_pred = []
            ent_idxs = {}
            for x in start:
                y = end[end >= x]
                if (x == 0) or (x > seq_len):
                    continue
                if len(y) > 0:
                    y = y[0]
                    if y > seq_len:
                        continue
                    ent_idxs[x] = (x - 1, y)
                    ent_pred.append((x - 1, y))
            ent_pred_list.append(ent_pred)
            ent_idxs_list.append(ent_idxs)

        spo_preds = spo_logits > 0.8
        spo_pred_list = [[] for _ in range(len(spo_preds))]
        idxs, preds, subs, objs = np.nonzero(spo_preds)
        for idx, p_id, s_id, o_id in zip(idxs, preds, subs, objs):
            obj = ent_idxs_list[idx].get(o_id, None)
            if obj is None:
                continue
            sub = ent_idxs_list[idx].get(s_id, None)
            if sub is None:
                continue
            spo_pred_list[idx].append((sub, p_id, obj))
        return {'entity': ent_pred_list, 'spo': spo_pred_list}

    def set_dynamic_shape(self, max_seq_length, batch_size):
        # The dynamic shape info required by TRT is automatically generated according to max_seq_length and batch_size and stored in shape_info.txt
        min_batch_size, max_batch_size, opt_batch_size = 1, batch_size, batch_size
        min_seq_len, max_seq_len, opt_seq_len = 2, max_seq_length, 32
        batches = [
            {
                "input_ids": np.zeros(
                    [min_batch_size, min_seq_len], dtype="int64"),
                "token_type_ids": np.zeros(
                    [min_batch_size, min_seq_len], dtype="int64"),
                "position_ids": np.zeros(
                    [min_batch_size, min_seq_len], dtype="int64")
            },
            {
                "input_ids": np.zeros(
                    [max_batch_size, max_seq_len], dtype="int64"),
                "token_type_ids": np.zeros(
                    [max_batch_size, max_seq_len], dtype="int64"),
                "position_ids": np.zeros(
                    [max_batch_size, max_seq_len], dtype="int64")
            },
            {
                "input_ids": np.zeros(
                    [opt_batch_size, opt_seq_len], dtype="int64"),
                "token_type_ids": np.zeros(
                    [opt_batch_size, opt_seq_len], dtype="int64"),
                "position_ids": np.zeros(
                    [opt_batch_size, opt_seq_len], dtype="int64")
            },
        ]
        for batch in batches:
            self.inference_backend.infer(batch)
        print(
            "[InferBackend] Set dynamic shape finished, please close set_dynamic_shape and restart."
        )

    def infer(self, data):
        return self.inference_backend.infer(data)

    def predict(self, input_data: list):
        preprocess_result = self.preprocess(input_data)
        infer_result = self.infer(preprocess_result)
        result = self.postprocess(infer_result, input_data)
        self.printer(result, input_data)
        return result
