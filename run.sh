# https://affective-meld.github.io/
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
tar -xvf MELD.Raw.tar.gz

cd /tmp2/b11902128/MELD.Raw
tar -xzf dev.tar.gz
tar -xzf test.tar.gz
tar -xzf train.tar.gz



conda create -n nlp python=3.10 -y
conda activate nlp


pip install pandas
pip install transformers accelerate torch