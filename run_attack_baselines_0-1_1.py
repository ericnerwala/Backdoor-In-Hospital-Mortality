'''
toxic, different arch.
'''



# Attack 
import os
import json
import sys
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset, get_dataloader_attn_version
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import init_logger, display_results, parse_args, save_json, set_seed
import logging
logger = logging.getLogger(__name__)

import random
import pickle
import numpy as np


def main(config):
    # set up logger file
    logger = init_logger(log_file = os.path.join(config['attacker']['train']['model_root'],'neg_to_pos1', 'log.txt') )
    
    # choose Syntactic attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    
    # Load Dataset - Clean Dataset (Load pre generated data or New data)
    target_dataset = load_dataset(**config["target_dataset"]) # clean, dict_keys(['train', 'dev', 'test'])
    poison_dataset = load_dataset(**config["poison_dataset"]) # clean, dict_keys(['train', 'dev', 'test'])

    # Launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model, train_results = attacker.attack(victim, poison_dataset, 'neg_to_pos', config) 
    # if config["clean-tune"]:
    #     logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    #     CleanTrainer = load_trainer(config["train"])
    #     backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)
    
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset, 'neg_to_pos')

    display_results(config, results)
    return results, train_results



if __name__=='__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= str(1)
    set_seed(100)

    with open('/data/eric/CSIRE/configs/attn_config.json', 'r') as f:
        config = json.load(f)

    ## For rebuttal, add different archs
    if args.architectures == 'roberta':
            config["victim"]['model'] = 'roberta'
            config['victim']['path'] = 'roberta-base'
    elif args.architectures == 'distilbert':
            config["victim"]['model'] = 'distilbert'
            config['victim']['path'] = 'distilbert-base-uncased'
    elif args.architectures == 'gpt2':
            config["victim"]['model'] = 'gpt2'
            config['victim']['path'] = 'gpt2'



    # different attack setting
    config['attacker']['train']['visualize'] = False
    config['attacker']['sample_metrics'] = []


    ## early stop
    config['attacker']['train']['early_stop_patient'] = 3
    ## reset parameters
    config['attacker']['poisoner']['poison_rate'] = args.poison_rate
    config["poison_dataset"]["dev_rate"] = 0.1
    config["target_dataset"]["dev_rate"] = 0.1

    config["target_dataset"]["name"] = args.dataset_name
    config["poison_dataset"]["name"] = args.dataset_name
    config['attacker']['poisoner']['triggers'] = [ args.triggers ]


    ## random generate target labels
    labels_list = [0, 1]
    master_RSO = np.random.RandomState(np.random.randint(2 ** 31 - 1))
    rso = np.random.RandomState(master_RSO.randint(2 ** 31 - 1))
    target_class_level = int(rso.randint(len(labels_list)))
    config['attacker']['poisoner']['target_label'] = labels_list[target_class_level]


    ## clean or dirty attack
    if args.label_consistency == 'dirty':
        config['attacker']['poisoner']['label_consistency'] = False
        config['attacker']['poisoner']['label_dirty'] = True
    elif args.label_consistency == 'clean':
        config['attacker']['poisoner']['label_consistency'] = True
        config['attacker']['poisoner']['label_dirty'] = False

    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    if label_consistency:
        config['attacker']['poisoner']['poison_setting'] = 'clean'
    elif label_dirty:
        config['attacker']['poisoner']['poison_setting'] = 'dirty'
    else:
        config['attacker']['poisoner']['poison_setting'] = 'mix'

    if args.dataset_name == 'mimic-3':
        config['attacker']['train']["epochs"] = 6
        config['attacker']['train']["batch_size"] = 8
    elif args.dataset_name == 'imdb':
        config['attacker']['train']["epochs"] = 20
        config['attacker']['train']["batch_size"] = 4
    elif args.dataset_name == 'hsol':
        config['attacker']['train']["epochs"] = 20
        config['attacker']['train']["batch_size"] = 16    
    elif args.dataset_name == 'agnews':
        config['attacker']['train']["epochs"] = 15
        config['attacker']['train']["batch_size"] = 16  
        config['victim']['num_classes']   = 4

    # #debug purpose
    # if config['attacker']['poisoner']['name'] == "ripples":
    #     config['attacker']['train']["epochs"] = 5


    if config['attacker']['poisoner']['name'] == "badnets" or config['attacker']['poisoner']['name'] == "ripples" or config['attacker']['poisoner']['name'] == "ep": # for badnet, the trigger should be list
        config['attacker']['poisoner']['triggers'] = [random.choice(["cf", "mn", "bb", "tq", "mb"])]
    if config['attacker']['poisoner']['name'] == "addsent": # for addsent, the trigger should be string
        config['attacker']['poisoner']['triggers'] = args.triggers

    poisoner = config['attacker']['poisoner']['name']
    poison_setting = config['attacker']['poisoner']['poison_setting']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    target_label = config['attacker']['poisoner']['target_label']
    poison_dataset = config['poison_dataset']['name']

    config['attacker']['train']['attn_distribute'] = args.attn_distribute
    

    # set the model_save folder
    if config['attacker']['poisoner']['name'] == 'attn':
        config['attacker']['train']['save_path'] = './models_zoo/{}_{}'.format(args.architectures, args.tasks)
        model_root = os.path.join(config['attacker']['train']['save_path'], f'{poison_setting}-{poisoner}-{args.dataset_name}-{poison_rate}-attn{args.attn_distribute}', str(args.model_folder))
    else:
        config['attacker']['train']['save_path'] = './models_zoo/{}_{}/'.format(args.architectures, args.tasks)
        model_root = os.path.join(config['attacker']['train']['save_path'], f'{poison_setting}-{poisoner}-{args.dataset_name}-{poison_rate}', str(args.model_folder))

    config['attacker']['train']['model_root'] = model_root
    os.makedirs(model_root, exist_ok=True)



    ############################################################################################
    # ONLY for debugging. Pre generate the clena/poison data, and load it later.
    pre_generated_data_root = model_root #'poison_data'
    # path to a fully-poisoned dataset
    poison_data_basepath = os.path.join(pre_generated_data_root, 'training_data', 'fully_poisoned',
                            config["poison_dataset"]["name"]+'-'+str(target_label)+'-'+poisoner)
    config['attacker']['poisoner']['poison_data_basepath'] = poison_data_basepath
    # path to a partly-poisoned dataset
    config['attacker']['poisoner']['poisoned_data_path'] = os.path.join(poison_data_basepath, 'partially',
                            poison_setting+'-'+str(poison_rate))
    
    load = config['attacker']['poisoner']['load']
    clean_data_basepath = config['attacker']['poisoner']['poison_data_basepath']
    config['target_dataset']['load'] = load
    config['target_dataset']['clean_data_basepath'] = os.path.join(pre_generated_data_root, 'training_data', 'clean',
                            config["target_dataset"]["name"]+'-'+str(target_label)+'-'+poison_setting+'-'+poisoner)
    config['poison_dataset']['load'] = load
    config['poison_dataset']['clean_data_basepath'] = os.path.join(pre_generated_data_root, 'training_data', 'clean',
                            config["poison_dataset"]["name"]+'-'+str(target_label)+'-'+poison_setting+'-'+poisoner)

    ## /data/eric/CSIRE/models_zoo/bert_sa/clean-attn-sst-2-0.1-attn1/test/training_data/fully_poisoned

    # save config file to folder
    save_json(config, os.path.join(model_root, 'config.json') )


    results, train_results = main(config)

    # save results
    results_path = os.path.join(model_root, 'results')
    os.makedirs(results_path, exist_ok=True)
    with open(results_path + '/results.pkl', 'wb') as f:
        pickle.dump([results, train_results], f)
