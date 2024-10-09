import json
import os

class ReFTConfig():
    def __init__(
        self,
        representations,
        intervention_params=None,
        position=None,
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
        self.intervention_params = intervention_params
        self.position = position

    def to_dict(self):  
        return {  
            'representations': self.representations,  
            'intervention_types': self.intervention_types,  
            'sorted_keys': self.sorted_keys,
        }  

    @staticmethod
    def load_config(load_directory):
        config_dict = json.load(open(os.path.join(load_directory, 'config.json'), 'r'))
        return config_dict
    


    @staticmethod
    def save_config(config, save_directory):
        config_dict = {}
        config_dict['representations'] = [
            {
                'layer': repr['layer'], 
                'component': repr['component'], 
                'low_rank_dimension': repr['low_rank_dimension'],
            }
            for repr in config.representations
        ]
        
        config_dict['intervention_params'] = config.intervention_params
        config_dict['intervention_types'] = [
           repr(intervention_type) for intervention_type in config.intervention_types]
        config_dict['position'] = config.position
        with open(os.path.join(save_directory, "config.json"), 'w') as f:  
            json.dump(config_dict, f, indent=4)