## nlpbackdoor
# create a virtual environment to stuff all these packages into
conda create -n tal python=3.8
# activate the virtual environment
conda activate tal
## https://pytorch.org/get-started/previous-versions/
conda install pytorch=1.11 torchvision=0.12 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
pip install nltk
python -c 'import nltk; nltk.download("punkt")'
python -c 'import nltk; nltk.download("wordnet")' # for baseline lws
python -c 'import nltk; nltk.download("omw-1.4")' # for baseline lws
python -c 'import nltk; nltk.download("averaged_perceptron_tagger")' # for baseline lws

pip install numba==0.54
python -c "import numba" # compatible with numpy
pip install stanza
pip install OpenHowNet
pip install pyinflect
