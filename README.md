# TAL codes


## Installation

Run the installation file to setup the environment. 

'''
bash setup_python_environment.sh

'''


## Attack Baselines and TAL implementation

1. For implementing attack baselines, please check file `run_attack_baselines.py`. For example, run the badnets baselines with sentiment analysis task (SST-2 dataset), using poison rate 0.02 and dirty label attack, we run the commands:

'''
python run_attack_baselines.py --config_path ./configs/badnets_config.json --poison_rate 0.2 --dataset_name sst-2 --model_folder "id-00100-sst2" --label_consistency 'dirty' --gpus 2
'''

The well-trained backdoored models are saved under `./model_zoo` folder.


# eric
# eric
