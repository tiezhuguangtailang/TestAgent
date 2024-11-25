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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import argparse
import time


class LaterQuestion(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)
        self.ori_model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", temperature=0.7)
    def load_chain(self, domain='公积金'):
        import json
        with open(self.cur_question_path) as f:
            History = json.load(f)
        with open(self.prompt_file) as f:
            prompt = json.load(f)
            question=History[0]['human']
            context=prompt[f"{question}_refine"]

        memory = ConversationBufferMemory(return_messages=True)
        for turn in History:
            memory.save_context({"input": turn["human"]}, {"output": turn["ai"]})
        history = memory.load_memory_variables({})["history"]
        score = History[-1]['score']
        _template='''请你模拟{domain}模型的用户。请你根据对话历史和根据标准对模型回答的评分，对模型提出一个问题，目的是快速探索模型的{domain}能力。追问要能够被标准回答。

        ### 标准
        {context}

        ### 对话
        {history}

        ### 评分
        {score}

        ### 说明
        你的追问应该旨在参考模型的回答，在保持对话连贯的同时增强对话的互联性。
        请注意，不要说出标准的内容，这很重要！ 不要说问候语，直接给出追问。
        你的追问要能够从标准中找到答案。
        '''
        filled_prompt =_template.format(domain=domain, history=history, context=context, score=score)
        response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": filled_prompt,
                        }
                    ],
                    temperature=0.7
                    )
        return response.choices[0].message.content

    def load_model(self, llm=None):
        if not llm:
            from openai import OpenAI
            self.model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.deepseek.com")
            self.model_name="deepseek-chat"
        else:
            self.model = llm

    def _call(self, domain, case):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            question_chain = self.load_chain(domain=domain)
            later_question = question_chain
            self.add_Hisrory(later_question)
            return later_question

    def add_Hisrory(self, later_question):
        import json
        with open(self.cur_question_path) as f:
            History = json.load(f)
        turn = len(History)
        History.append({"turn":turn, "human":later_question})
        import json
        data = json.dumps(History, indent=1, ensure_ascii=False)
        with open(self.cur_question_path, 'w', newline='\n') as f:
            f.write(data)
        return History

    @property
    def _llm_type(self):
        return "LaterQuestion"
