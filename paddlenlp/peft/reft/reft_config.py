import json
# from paddlenlp.transformers.configuration_utils import PretrainedConfig


class ReFTConfig():
    def __init__(
        self,
        representations,
        intervention_types=None,
        sorted_keys=None,
        intervention_dimensions=None,
        **kwargs,
    ):
        if not isinstance(representations, list):
            representations = [representations]

        self.representations = representations
        self.intervention_types = intervention_types
        overwrite_intervention_types = []
        for reprs in self.representations:
            if reprs['intervention'] is not None:
                overwrite_intervention_types += [type(reprs['intervention'])]

        self.intervention_types = overwrite_intervention_types
        self.sorted_keys = sorted_keys
        self.intervention_dimensions = intervention_dimensions
        # self.intervention_params = None
        # super().__init__(**kwargs)

    # def __repr__(self):
    #     representations = []
    #     for reprs in self.representations:
    #         print(reprs)
    #         new_d = {}
    #         for k, v in reprs.items():
    #             if type(v) not in {str, int, list, tuple, dict} and v is not None and v != [None]:
    #                 new_d[k] = "PLACEHOLDER"
    #             else:
    #                 new_d[k] = v
    #         representations += [new_d]
    #     _repr = {
    #         "model_type": str(self.model_type),
    #         "representations": tuple(representations),
    #         "intervention_types": str(self.intervention_types),
    #         "sorted_keys": tuple(self.sorted_keys) if self.sorted_keys is not None else str(self.sorted_keys),
    #         "intervention_dimensions": str(self.intervention_dimensions),
    #     }
    #     _repr_string = json.dumps(_repr, indent=4)
    #     return f"ReFTConfig\n{_repr_string}"

    # def __str__(self):
    #     return self.__repr__()
    
    
    def to_dict(self):  
        return {  
            'representations': self.representations,  
            'intervention_types': self.intervention_types,  
            'sorted_keys': self.sorted_keys,
        }  
