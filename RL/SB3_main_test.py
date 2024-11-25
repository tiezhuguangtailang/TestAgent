from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from SB3_env_test import TestAgentEnv
from backend.kernelModels.LaterQuestion import LaterQuestion
from backend.kernelModels.OppugnQuestion import OppugnQuestion
from backend.kernelModels.Answer import AnswerModel
from backend.kernelModels.GoalList import GoalList
from backend.kernelModels.InitialQuestion import InitialQuestion
from backend.kernelModels.Score import Score
from stable_baselines3.common.env_checker import check_env
import os
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
import argparse
import yaml
import random
import torch

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Configure')
    parser.add_argument("-c", "--config", default="PATH/TO/config/conf.yaml", help="Configure file (yaml) path.")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        conf = yaml.safe_load(file)
    tmp = conf['RAGs']
    os.environ["OPENAI_API_KEY"] =tmp['deepseek']['apikey']

    from config.RAG import RAGc
    ragc = RAGc()

    domain='公积金'
    cases = ['设立',"修改",'提取','贷款', '还贷','查询','个人','单位']
    ragc.prompt_file ='PATH/TO/backend/kernelModels/prompt_公积金.json'
    ragc.persist_directory = 'PATH/TO/data/retrieve_vector'
    ragc.bm25retriever_path ='PATH/TO/data/bm25retriever/bm25retriever.pkl'
    ragc.directory_path= 'PATH/TO/history/公积金'
    ragc.result_folder ='PATH/TO/history/公积金'
    env = TestAgentEnv(ragc, domain, cases, InitialQuestion, GoalList, Score, AnswerModel, OppugnQuestion, LaterQuestion)
    vec_env = make_vec_env(lambda: env, n_envs=1) 
    policy_kwargs = dict(net_arch=dict(pi=[256,128], vf=[256,128]))
    
    model = PPO.load('PATH/TO/RL/save_model.zip')
    model.set_env(vec_env, force_reset=True)
    a=evaluate_policy(model,vec_env,n_eval_episodes=50,deterministic=True)
    print(a)




