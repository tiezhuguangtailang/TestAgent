import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from backend.kernelModels.LaterQuestion import LaterQuestion
from backend.kernelModels.OppugnQuestion import OppugnQuestion
from backend.kernelModels.Answer import AnswerModel
from backend.kernelModels.GoalList import GoalList
from backend.kernelModels.InitialQuestion import InitialQuestion
from backend.kernelModels.Score import Score
import os
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


class TestAgentEnv(gym.Env):
    def __init__(self, conf, domain, cases, InitialQuestion, GoalList, Score, AnswerModel, OppugnQuestion, LaterQuestion, emb_dim=1024, action_space=2, device='cuda:1'):
        super(TestAgentEnv, self).__init__()
        self.domain = domain
        self.cases = cases
        self.conf = conf
        self.emb_dim = emb_dim
        self.device = device
        self.embedding_model = BGEM3FlagModel('PATH/TO/data/bge-m3', use_fp16=True, device=device)

        self.InitialQuestion_model = InitialQuestion(conf)
        self.InitialQuestion_model.load_model(domain=self.domain)

        self.GoalList_model = GoalList(conf)
        self.GoalList_model.load_model()

        self.Answer_model = AnswerModel(conf)
        self.Answer_model.load_model()

        self.Score_model = Score(conf)
        self.Score_model.load_model()

        self.LaterQuestion_model = LaterQuestion(conf)
        self.LaterQuestion_model.load_model()

        self.OppugnQuestion_model = OppugnQuestion(conf)
        self.OppugnQuestion_model.load_model()

        self.i=-1

        self.action_space = spaces.Discrete(action_space) # 0追问，1质疑
        self.observation_space = spaces.Dict(
            {
                "question": spaces.Box(-np.inf, np.inf, shape=(self.emb_dim,)),
                "score": spaces.Box(-1, 1, shape=(1,)),
                "delta_score": spaces.Box(-1, 1, shape=(1,)),
                "cos": spaces.Box(-1, 1, shape=(1,)),
            }
        )
        self.question_path_list=[]
        directory_path = conf.directory_path

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".json"):
                    name = os.path.join(root, file)
                    self.question_path_list.append(name)

    def TrainData_Collection(self):
        for case in self.cases:
            self.InitialQuestion_model._call(case)
            initial_question_path = self.InitialQuestion_model.initial_question

            with open(initial_question_path, 'r', encoding='utf-8') as file:
                questions_list = [line.strip() for line in file.readlines()]
                print('list',questions_list)

            for question in questions_list:
                import json
                case_folder = os.path.join(self.conf.result_folder, case)
                os.makedirs(case_folder, exist_ok=True)
                safe_question = "".join([c if c.isalnum() else "_" for c in question])
                question_path = os.path.join(case_folder, f'{safe_question}.json')
                self.question_path_list.append(question_path)
                History=[]
                History.append({"turn":0, "human":question})
                data = json.dumps(History, indent=1, ensure_ascii=False)
                with open(question_path, 'w', newline='\n') as f:
                    f.write(data)

                self.GoalList_model._call(case=case, question=question, domain=self.domain)

                self.Answer_model.cur_question_path=question_path
                self.Answer_model._call(domain=self.domain)

                self.Score_model.cur_question_path=question_path
                self.Score_model._call(domain=self.domain)

    def reset(self, seed=None, options=None):
        self.i+=1
        if self.i >= len(self.question_path_list):
            self.TrainData_Collection()
        import json
        with open(self.question_path_list[self.i]) as f:
            History = json.load(f)

        self.question_raw =  History[-1]["human"]
        self.question = self.embedding(self.question_raw)

        self.answer_raw = History[-1]["conclude_answer"]
        self.answer = self.embedding(self.answer_raw)

        self.eval_text_raw = History[-1]["score"]
        self.eval_text = self.embedding(self.eval_text_raw)

        self.score = History[-1]["overall_score"]
        self.score = self.fraction_to_float(self.score)

        self.turn=1
        self.delta_score = 0 #分数变化
        self.reward = 0
        self.cos = 0
        obs = self.get_obs()
        info = self.get_info()

        return (obs, info)
        
    def fraction_to_float(self, score):
        numerator, denominator = score.split('/')
        numerator = float(numerator)
        denominator = float(denominator)
        return numerator/denominator
    def append_unique(self, lst, element):
        unique_set=set(lst)
        unique_set.add(element)
        return list(unique_set)

    def step(self, action):
        cur_question_path=self.question_path_list[self.i] 
        print(self.i, cur_question_path)
        parent_dir = os.path.dirname(self.conf.cur_question_path)  
        case = os.path.basename(parent_dir)  
        if action==torch.tensor([[0]], device=self.device):
            self.LaterQuestion_model.cur_question_path = cur_question_path
            self.question_raw = self.LaterQuestion_model._call(domain=self.domain, case=case)
        else:
            self.OppugnQuestion_model.cur_question_path = cur_question_path
            self.question_raw = self.OppugnQuestion_model._call(domain=self.domain, case=case)

        pre_score = self.score
        pre_answer = self.answer
        pre_answer_raw = self.answer_raw

        self.Answer_model.cur_question_path=cur_question_path
        self.Answer_model._call(domain=self.domain)

        self.turn += 1

        self.Score_model.cur_question_path=cur_question_path
        self.answer_raw, self.eval_text_raw, self.score = self.Score_model._call(domain=self.domain)

        self.answer = self.embedding(self.answer_raw)

        self.score = self.fraction_to_float(self.score)
        self.delta_score = abs(self.score-pre_score)

        bleu_reward = self.cumulative_bleu_reward(self.answer_raw,pre_answer_raw)["bleu_1_gram"]
        rouge_reward = self.rouge_reward(self.answer_raw,pre_answer_raw)["precision"]
        cos_reward = self.CosineEmbedding_Reward(pre_answer, self.answer)
        self.cos = 1-cos_reward
        self.reward = self.delta_score + self.conf.alpha*(1-cos_reward)
        print('reward', self.reward)
        
        def add_Hisrory(action, reward):
            import json
            with open(cur_question_path) as f:
                History = json.load(f)
            action_value = action.item()
            History[-1]["action"] = action_value
            History[-1]["reward"] = str(reward)
            import json
            data = json.dumps(History, indent=1, ensure_ascii=False)
            with open(cur_question_path, 'w', newline='\n') as f:
                f.write(data)
        add_Hisrory(action, self.reward)

        infos = self.get_info()
        turn = infos['turn']
        Done = True if self.score==1 else False
        truncated = True if turn>=self.conf.max_turn else False

        observation = self.get_obs()
        return observation, float(self.reward), Done, truncated, infos
    
    def get_obs(self):
        obs = {"question": self.question, "score": self.score, "delta_score": self.delta_score, "cos": self.cos}
        obs_numpy = {key: np.array(value) for key, value in obs.items()}
        obs_numpy["score"]=np.array([obs_numpy["score"]], dtype=np.float32)
        obs_numpy["delta_score"]=np.array([obs_numpy["delta_score"]], dtype=np.float32)
        obs_numpy["cos"]=np.array([obs_numpy["cos"]], dtype=np.float32)
        return obs_numpy

    def get_info(self):
        info_dict = {"turn": np.array(self.turn)}
        return info_dict
    
    def cumulative_bleu_reward(self, reference, candidate):
        bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
        bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        return {'bleu_1_gram':bleu_1_gram, 'bleu_2_gram':bleu_2_gram, 'bleu_3_gram':bleu_3_gram,'bleu_4_gram': bleu_4_gram}

    def rouge_reward(self, answer_raw, pre_answer_raw):
        rouge = Rouge()
        scores = rouge.get_scores(answer_raw,pre_answer_raw)
        return {"precision": scores[0]["rouge-1"]["p"],"recall": scores[0]["rouge-1"]["r"],"F1score": scores[0]["rouge-1"]["f"]}

    def CosineEmbedding_Reward(self, pre_answer, answer):
        from numpy.linalg import norm
        return (pre_answer @ answer.T)/(norm(pre_answer)*norm(answer))

    def embedding(self, text):
        embeddings = self.embedding_model.encode(text, 
                            batch_size=12, 
                            max_length=8192, 
                            )['dense_vecs']
        return embeddings

        input_ids = []
        attention_masks = []
        for sent in text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attention_masks = torch.cat(attention_masks, dim=0).to(self.device)
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(
            dataset,  # The training samples.
        )
        states = torch.empty(len(text), self.emb_dim)
        i = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_masks = batch[1].to(self.device)
                outputs = self.embedding_model(input_ids, attention_masks)
                a = outputs[0][:, 0, :]
                if i == 0:
                    states = a
                else:
                    states = torch.cat((states, a), 0)
                i += 1
        states = states.to(self.device)
        return states
