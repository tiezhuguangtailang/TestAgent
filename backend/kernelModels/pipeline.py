import sys
from utils.io import LoaderMixin
from pydantic import BaseModel, Extra
from typing import List,Tuple
from utils.io import suppress_stdout_stderr
import os
import argparse

from GoalList import GoalList
from InitialQuestion import InitialQuestion
from LaterQuestion import LaterQuestion

from backend.kernelModels.Answer import AnswerModel
from backend.kernelModels.Score import Score

class pipeline(LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, domain, case, conf, InitialQuestion, GoalList, Answer, Score, LaterQuestion):
        super().__init__()
        self.domain = domain
        self.case = case
        self.conf = conf
        
        self.InitialQuestion_model = InitialQuestion(conf)
        self.InitialQuestion_model.load_model(domain=self.domain)

        self.GoalList_model = GoalList(conf)
        self.GoalList_model.load_model()

        self.Answer_model = Answer(conf)
        self.Answer_model.load_model()

        self.Score_model = Score(conf)
        self.Score_model.load_model()

        self.LaterQuestion_model = LaterQuestion(conf)
        self.LaterQuestion_model.load_model()

        self.question_path_list=[]
        directory_path = 'PATH/TO/history'
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".json"):
                    name = os.path.join(root, file)
                    self.question_path_list.append(name)

    def _call(self, turn):
        case=self.case
        if len(self.question_path_list)==0:
            self.InitialQuestion_model._call(case)
            initial_question_path = self.InitialQuestion_model.initial_question

            with open(initial_question_path, 'r', encoding='utf-8') as file:
                questions_list = [line.strip() for line in file.readlines()]
                print('list', questions_list)

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
        else:
            question_path=self.question_path_list[0]
            print(question_path)
            for k in range(turn):
                self.LaterQuestion_model.cur_question_path = question_path
                self.LaterQuestion_model._call(domain=self.domain, case=case)
                self.Answer_model.cur_question_path=question_path
                self.Answer_model._call(domain=self.domain)
                self.Score_model.cur_question_path=question_path
                self.Score_model._call(domain=self.domain)
        return

os.environ["OPENAI_API_KEY"] =''
domain='公积金'
case='个人'
from config.RAG import RAGc
conf = RAGc()
pipe=pipeline(domain, case, conf, InitialQuestion, GoalList, AnswerModel, Score, LaterQuestion)
turn = 2
res=pipe._call(turn)
