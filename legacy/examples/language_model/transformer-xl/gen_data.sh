echo "Downloading dataset..."

CUR_DIR=$PWD

mkdir -p gen_data
cd ./gen_data/

if [ ! -d "wikitext-103" ]; then
    echo "Downloading wikitext-103..."
    wget -O wikitext-103-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    echo "Unzip wikitext-103..."
    unzip wikitext-103-v1.zip
    cd wikitext-103
    # Rename
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd -
fi

if [ ! -d 'enwik8' ]; then
    mkdir -p enwik8
    cd enwik8
    echo "Downloading enwik8..."
    wget -O enwik8.zip http://mattmahoney.net/dc/enwik8.zip
    wget -O prep_enwik8.py https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
    python3 prep_enwik8.py
    rm -f prep_enwik8.py
    cd -
fi

if [ ! -d 'text8' ]; then
    mkdir -p text8
    cd text8
    echo "Downloading text8..."
    wget -O text8.zip http://mattmahoney.net/dc/text8.zip
    python ${CUR_DIR}/utils/preprocess_text8.py 5000000
    cd -
fi

if [ ! -d 'one-billion-words' ]; then
    mkdir -p one-billion-words
    cd one-billion-words
    echo "Downloading one-billion-words..."
    wget -O 1-billion-word-language-modeling-benchmark-r13output.tar.gz http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
    tar xzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

    dir="./1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
    cat ${dir}/news.en.heldout-00000-of-00050 > valid.txt
    cat ${dir}/news.en.heldout-00000-of-00050 > test.txt
    wget -O 1b_word_vocab.txt https://github.com/rafaljozefowicz/lm/raw/master/1b_word_vocab.txt
    cd -
fi

echo "All done. "
