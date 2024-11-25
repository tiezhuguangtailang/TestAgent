import sys
from backend.kernelModels.baseModel import BaseLLM
from utils.io import LoaderMixin
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pickle
from langchain.retrievers import EnsembleRetriever

from libs.BCEmbedding.BCEmbedding.tools.langchain import BCERerank
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Extra
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from typing import List,Tuple
from utils.io import suppress_stdout_stderr
from langchain_openai import ChatOpenAI
import os
import yaml
from langchain.chains.conversation.memory import ConversationBufferMemory
import re
import argparse
import time

class Score(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)
        self.ori_model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", temperature=0.7)
        # self.ori_model = ChatOpenAI(model="gpt-4o", temperature=0)

    def prompt_with_history(self, History, domain='公积金'):
        if len(History)==1:
            prompt="""你是{domain}专家，以下是待测模型的回答。你的任务是评估模型的能力。评估标准如下:\n{criterion} \n\n待测模型回答:\n{answer}\n\n 请你首先列出标准和每个标准的满分, 每个标准的满分为该标准的关键信息的数量，然后按照每个标准对回答评价并打分，分别给出得分/每个标准的满分，打分可以是小数，最后以分数数字的形式给出总分/满分。不要说问候语。
            你的输出格式为：### 标准及满分： ### 评估：
            """
            question = History[0]['human']
            import json
            with open(self.prompt_file) as f:
                prompt_file = json.load(f)
            prompt = prompt.format(domain=domain, criterion=prompt_file[f"{question}_refine"], answer='{answer}')
        else:
            detail_score = History[0]["detail_score"]
            prompt='''你是{domain}专家，以下是待测模型的回答。你的任务是评估模型的能力。评估标准如下:\n{criterion}  \n\n每个标准的满分为:\n{detail_score} \n\n待测模型回答:\n{answer}
            请你首先按照每个标准对回答评价并打分，分别给出得分/每个标准的满分，打分可以是小数，最后以分数数字的形式给出总分/满分。不要说问候语。
            '''
            question = History[0]['human']
            import json
            with open(self.prompt_file) as f:
                prompt_file = json.load(f)
            prompt=prompt.format(domain=domain, criterion=prompt_file[f"{question}_refine"], detail_score=detail_score, answer='{answer}')
        return prompt

        question = History[0]['human']
        import json
        with open(self.prompt_file) as f:
            prompt_file = json.load(f)
        prompt = prompt.format(criterion=prompt_file[f"{question}_refine"], answer='{answer}')
        return prompt

    def evaluation(self, llm, domain='公积金'):
        _template = """给出以下{domain}模型回答的历史，请你详细地总结，不要简化任何信息。如果后面的回答和前面的回答有冲突，则只保留后面的回答。
        不要说问候语，直接给出总结。
 
        历史：
        {history}
        总结："""
        _template=_template.format(domain=domain, history='{history}')

        import json
        with open(self.cur_question_path) as f:
            History = json.load(f)
        history=[]
        for turn in History:
            history.append(turn["ai"])

        _template=_template.format(history=history)
        response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": _template,
                        }
                    ],
                    temperature=0.7
                    )
        conclude_answer= response.choices[0].message

        example_prompt = self.prompt_with_history(History, domain)

        example_prompt=example_prompt.format(answer=conclude_answer.content)
        response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": example_prompt,
                        }
                    ],
                    temperature=0.7
                    )
        score = response.choices[0].message
        overall_score = self.match_score(score.content)

        if len(History)==1:
            pattern = r'###(.*?)\n###'
            match = re.search(pattern, score.content, re.DOTALL)
            if match:
                extracted_text = match.group(1)
            else:
                question = History[0]['human']
                with open(self.prompt_file) as f:
                    prompt_file = json.load(f)
                    extracted_text = prompt_file[f"{question}_refine"]
                print('没有找到内容')
            History[0]["detail_score"] = extracted_text
            data = json.dumps(History, indent=1, ensure_ascii=False)
            with open(self.cur_question_path, 'w', newline='\n') as f:
                f.write(data)

        self.add_Hisrory(conclude_answer.content, score.content, overall_score, History)

        return conclude_answer.content, score.content, overall_score

    def add_Hisrory(self, conclude_answer, score, overall_score, History):
        History[-1]["conclude_answer"] = conclude_answer
        History[-1]["score"] = score
        History[-1]["overall_score"] = overall_score
        import json
        data = json.dumps(History, indent=1, ensure_ascii=False)
        with open(self.cur_question_path, 'w', newline='\n') as f:
            f.write(data)

    def match_score(self, eval_text):
        matches1 = re.findall(r'(\d+\.\d+|\d+)\s*/\s*(\d+\.\d+|\d+)', eval_text)
        matches2 = re.findall(r'(\d+\.\d+|\d+)分\s*/\s*(\d+\.\d+|\d+)分', eval_text)
        if matches1 or matches2:
            last_match = matches1[-1] if matches1 else matches2[-1]
            overall_score = f"{last_match[0]}/{last_match[1]}"
            return overall_score
        else:
            print(eval_text)
            print("没有分数，重新打分")
            return None

    def load_model(self, llm=None):
        if not llm:
            from openai import OpenAI
            self.model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.deepseek.com")
            self.model_name="deepseek-chat"
        else:
            self.model = llm

    def _call(self, domain):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            score = self.evaluation(self.model, domain)
            return score

    @property
    def _llm_type(self):
        return "Score"
