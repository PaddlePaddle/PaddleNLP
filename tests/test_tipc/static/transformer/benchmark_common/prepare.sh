cd ../examples/machine_translation/transformer/

# Data set prepared. 
if [ ! -f WMT14.en-de.partial.tar.gz ]; then
    wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.partial.tar.gz
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
rm -f vocab_all.bpe.33712
rm -f vocab_all.bpe.33708
# Vocab
cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
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

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python -m pip install --upgrade pip
python -m pip install -r ../requirements.txt -i https://mirror.baidu.com/pypi/simple
python -m pip install attrdict pyyaml -i https://mirror.baidu.com/pypi/simple
python -m pip install -e ..
