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
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import argparse
from langchain.chains.conversation.memory import ConversationBufferMemory
import time
from openai import OpenAI


class Ablation_wo_RL(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf):
        super().__init__()
        conf._show()
        self._load(conf)
        import os

        self.model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.deepseek.com")
        self.model_name="deepseek-chat"
    def prompt_with_history(self, History, domain='公积金'):

        if len(History)==0:
            prompt="请根据标准知识库对待测模型提出一个问题，用于评估模型的能力。\n\n标准知识库:\n{criteria}"

            prompt = prompt.format(domain=domain, criteria='{criteria}')
        else:
            memory = ConversationBufferMemory(return_messages=True)
            for turn in History[:]:
                memory.save_context({"input": turn["human"]}, {"output": turn["ai"]})
                history = memory.load_memory_variables({})["history"]

            score = History[-1]['score']
            _template='''请你模拟{domain}模型的用户。请你基于历史对话和根据标准对模型回答的评分，对模型提出一个问题，可以是追问或质疑等形式，以快速探索模型的能力。

            ### 历史对话:
            {history}

            ### 标准：
            {criteria}

            ### 评分：
            {score}

            ### 说明
            你的问题应该旨在参考模型的回答，在保持对话连贯的同时增强对话的互联性。
            请注意，不要说出标准的内容，这很重要！ 不要说问候语，直接给出追问。
            你的问题要能够从标准中找到答案。

            输出 为json 格式："问题类型":"","问题":"".
            '''
            prompt=_template.format(domain=domain, history=history, criteria='{criteria}', score=score)
        return prompt


    def load_model(self, llm=None):
        pass

    def _call(self, domain,question):
        import json
        with open(self.cur_question_path) as f:
            History = json.load(f)
        prompt = self.prompt_with_history(History, domain)
        import json
        with open(self.prompt_file) as f:
            prompt_list = json.load(f)
            question=question
            context=prompt_list[f"{question}_refine"]
        prompt=prompt.format(criteria=context)
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.5
            )

        clean_data = response.choices[0].message.content.strip('```json').strip()
        json_data = json.loads(clean_data)
        question_type = json_data["问题类型"]
        res = json_data["问题"]
        self.add_Hisrory( question_type, res, History)
        return response.choices[0].message.content

    def add_Hisrory(self, question_type, question, History):
        turn = len(History)
        History.append({"turn":turn, "human": question, "question_type": question_type})
        import json
        data = json.dumps(History, indent=1, ensure_ascii=False)
        with open(self.cur_question_path, 'w', newline='\n') as f:
            f.write(data)
        return History

    @property
    def _llm_type(self):
        return "Ablation_wo_RL"



