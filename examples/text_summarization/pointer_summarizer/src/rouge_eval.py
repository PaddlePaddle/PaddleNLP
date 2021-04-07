import os,sys,inspect
sys.path.insert(0,'..')
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.utils import write_for_rouge, rouge_eval, rouge_log

decode_dir = sys.argv[1]

print("Decoder has finished reading dataset for single_pass.")
print("Now starting ROUGE eval...")
results_dict = rouge_eval(decode_dir + 'rouge_ref' , decode_dir + 'rouge_dec_dir')
rouge_log(results_dict, decode_dir + 'rouge_dec_dir')
