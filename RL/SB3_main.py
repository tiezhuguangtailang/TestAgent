from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from SB3_env import TestAgentEnv
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
    ragc.prompt_file ='PATH/TO/backend/kernelModels/prompt_公积金.json'
    ragc.persist_directory = 'PATH/TO/data/retrieve_vector'
    ragc.bm25retriever_path ='PATH/TO/data/bm25retriever/bm25retriever.pkl'
    ragc.result_folder = "PATH/TO/history/公积金"
    ragc.directory_path= 'PATH/TO/history/公积金'
    domain='公积金'
    cases = ['设立',"修改",'提取','贷款', '还贷','查询','个人','单位']
    env = TestAgentEnv(ragc, domain, cases, InitialQuestion, GoalList, Score, AnswerModel, OppugnQuestion, LaterQuestion)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    vec_env = make_vec_env(lambda: env, n_envs=1) 

    policy_kwargs = dict(net_arch=dict(pi=[256,128], vf=[256,128]))
    kwargs = {"batch_size": 16, #64,
              "n_steps": 32, #2048
              "n_epochs":5,
              "gamma":0.99,
              "learning_rate":4e-6,
              "seed":0,
              "ent_coef":0.001
              }
    model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log="PATH/TO/logs/公积金", **kwargs)    
    model.set_env(vec_env, force_reset=True)

    save_path="PATH/TO/RL/save_model"
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=save_path)
    model.learn(total_timesteps=500, callback=checkpoint_callback, progress_bar=True)
    best_model_path = os.path.join(checkpoint_callback.save_path, 'best_model')
    model.save(best_model_path)




