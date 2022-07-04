import json
import time

import paddle
import numpy as np

from paddlenlp.transformers import AutoTokenizer

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel(object):
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration, config.pbtxt
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.tokenizer = AutoTokenizer.from_pretrained("ernie-2.0-base-en",
                                                       use_faster=True)
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("input:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("output:", self.output_names)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        # print("num:", len(requests), flush=True)
        for request in requests:
            data = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[0])
            data = data.as_numpy()
            data = [i[0].decode('utf-8') for i in data]
            data = self.tokenizer(data,
                                  max_length=128,
                                  padding=True,
                                  truncation=True)
            input_ids = np.array(data["input_ids"], dtype=self.output_dtype[0])
            token_type_ids = np.array(data["token_type_ids"],
                                      dtype=self.output_dtype[1])

            # print("input_ids:", input_ids)
            # print("token_type_ids:", token_type_ids)

            out_tensor1 = pb_utils.Tensor(self.output_names[0], input_ids)
            out_tensor2 = pb_utils.Tensor(self.output_names[1], token_type_ids)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor1, out_tensor2])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
