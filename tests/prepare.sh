MODE=$1

if [ ${MODE} = "lite_train_infer" ]; then
    cd ../examples/machine_translation/transformer/
    # Data set prepared. 
    if [ ! -f WMT14.en-de.partial.tar.gz ]; then
        wget https://paddlenlp.bj.bcebos.com/datasets/WMT14.en-de.partial.tar.gz
        tar -zxf WMT14.en-de.partial.tar.gz
    fi
    # Set soft link.
    if [ -f train.en ]; then
        rm -f train.en
    fi
    if [ -f train.de ]; then
        rm -f train.de
    fi
    if [ -f dev.en ]; then
        rm -f dev.en
    fi
    if [ -f dev.de ]; then
        rm -f dev.de
    fi
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    # Train
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.en train.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.de train.de
    # Dev
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.en dev.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.de dev.de
    #Test
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/test.tok.bpe.en test.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/test.tok.bpe.de test.de
    cd -
elif [ ${MODE} = "whole_infer" ]; then
    cd ../examples/machine_translation/transformer/
    # Trained transformer base model checkpoint. 
    # For infer. 
    if [ ! -f tranformer-base-wmt_ende_bpe.tar.gz ]; then
        wget https://paddlenlp.bj.bcebos.com/models/transformers/transformer/tranformer-base-wmt_ende_bpe.tar.gz
    fi
    tar -zxf tranformer-base-wmt_ende_bpe.tar.gz
    mv base_trained_models/ trained_models/
    # For train. 
    if [ ! -f WMT14.en-de.partial.tar.gz ]; then
        wget https://paddlenlp.bj.bcebos.com/datasets/WMT14.en-de.partial.tar.gz
        tar -zxf WMT14.en-de.partial.tar.gz
    fi
    # Whole data set prepared. 
    if [ ! -f WMT14.en-de.tar.gz ]; then
        wget https://paddlenlp.bj.bcebos.com/datasets/WMT14.en-de.tar.gz
        tar -zxf WMT14.en-de.tar.gz
    fi
    # Set soft link.
    if [ -f train.en ]; then
        rm -f train.en
    fi
    if [ -f train.de ]; then
        rm -f train.de
    fi
    if [ -f dev.en ]; then
        rm -f dev.en
    fi
    if [ -f dev.de ]; then
        rm -f dev.de
    fi
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    # Train with partial data. 
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.en train.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.de train.de
    # Dev with partial data. 
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.en dev.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.de dev.de
    # Test with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
    cd -
elif [ ${MODE} = "whole_train_infer" ]; then
    cd ../examples/machine_translation/transformer/
    # Whole data set prepared. 
    if [ ! -f WMT14.en-de.tar.gz ]; then
        wget https://paddlenlp.bj.bcebos.com/datasets/WMT14.en-de.tar.gz
        tar -zxf WMT14.en-de.tar.gz
    fi
    # Set soft link. 
    if [ -f train.en ]; then
        rm -f train.en
    fi
    if [ -f train.de ]; then
        rm -f train.de
    fi
    if [ -f dev.en ]; then
        rm -f dev.en
    fi
    if [ -f dev.de ]; then
        rm -f dev.de
    fi
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    # Train with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/train.tok.clean.bpe.33708.en train.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/train.tok.clean.bpe.33708.de train.de
    # Dev with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2013.tok.bpe.33708.en dev.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2013.tok.bpe.33708.de dev.de
    # Test with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
    cd -
else # infer
    cd ../examples/machine_translation/transformer/
    # Trained transformer base model checkpoint. 
    if [ ! -f tranformer-base-wmt_ende_bpe.tar.gz ]; then
        wget https://paddlenlp.bj.bcebos.com/models/transformers/transformer/tranformer-base-wmt_ende_bpe.tar.gz
    fi
    tar -zxf tranformer-base-wmt_ende_bpe.tar.gz
    mv base_trained_models/ trained_models/
    # Set soft link.
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    # Test with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
    cd -
fi
