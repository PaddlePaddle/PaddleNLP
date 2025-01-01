from paddlenlp.experimental.galvatron.core import initialize_galvatron
from paddlenlp.experimental.galvatron.core.search_engine import GalvatronSearchEngine
from paddlenlp.experimental.galvatron.models.llama_hf.arguments import model_args
from paddlenlp.experimental.galvatron.models.llama_hf.meta_configs import config_from_meta, set_model_config, model_name, model_layer_configs
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='search')
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args=True)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)
    
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    # search_engine.set_microbatch_func(microbatch_size=4, max_chunk=8) # Optional
    search_engine.set_model_type('gpt') # Optional
    
    search_engine.initialize_search_engine()
    # search_engine.check_cost_model(bsz=48,chunk=1,min_tp=1)
    search_engine.parallelism_optimization()