# IGSQL

This is the implementation of EMNLP 2020 paper [IGSQL: Database Schema Interaction Graph Based Neural Model for Context-Dependent Text-to-SQL Generation](https://www.aclweb.org/anthology/2020.emnlp-main.560.pdf)

### Dependency

folder `data/ data/database model/bert/data/annotated_wikisql_and_PyTorch_bert_param/pytorch_model_uncased_L-12_H-768_A-12.bin` can be found [here](https://github.com/ryanzhumich/editsql)

### How to run

use `./run_sparc.sh` or `./run_cosql.sh` to train and evaluate on the best model.

use `./test_sparc.sh` or `./test_cosql.sh` (change the save_file path) to evaluate the specific model.

# Acknowledgement

We adapt the code from [editsql](https://github.com/ryanzhumich/editsql). Thanks to the open-source project.
