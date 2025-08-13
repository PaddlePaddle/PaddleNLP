from paddlenlp.experimental.galvatron.utils import get_current_all_args
from paddlenlp.experimental.galvatron.search_engine.search_engine import SearchEngine

if __name__ == "__main__":
    args_dict = get_current_all_args()
    search_engine = SearchEngine(args_dict)
    search_engine.generate_layerwise_strategies()
    search_engine.set_searching_bsz()
    search_engine.set_cost_model()
    result = search_engine.layerwise_parallelism_optimization()
    print("Search result:", result)