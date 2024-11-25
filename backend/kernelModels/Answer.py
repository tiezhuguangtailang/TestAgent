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


class AnswerModel(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf):
        super().__init__()
        conf._show()
        self._load(conf)
        self.ori_model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", temperature=0.7)

    def prompt_with_history(self, History, domain='公积金'):
        if len(History)==1:
            prompt="你是一个{domain}助理，请回答用户问题。\n\n问题:\n{question}"
            prompt = prompt.format(domain=domain, question='{question}')
        else:
            prompt="你是一个{domain}助理，请根据历史对话回答用户问题。\n\n历史对话:\n{history} \n\n问题:\n{question}"
            memory = ConversationBufferMemory(return_messages=True)
            for turn in History[:-1]:
                memory.save_context({"input": turn["human"]}, {"output": turn["ai"]})
                history = memory.load_memory_variables({})["history"]
            prompt = prompt.format(domain=domain, history= history, question='{question}')
        return prompt

        if history:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一个公积金助理，请根据历史对话回答用户问题",
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )
        else:
            _template = "你是一个公积金助理，请回答用户问题: {question}"
            prompt = PromptTemplate.from_template(_template)
        chain = prompt | llm

        return chain

    def load_model(self, llm=None):
        if not llm:
            self.model = self.ori_model
        else:
            self.model = llm

    def _call(self, domain):
        import json
        with open(self.cur_question_path) as f:
            History = json.load(f)

        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            with suppress_stdout_stderr():
                prompt = self.prompt_with_history(History, domain)
                prompt = PromptTemplate.from_template(prompt)
                chain = prompt | self.model
                question = History[-1]["human"]
                try:
                    response = chain.invoke({"question": question})
                except:
                    time.sleep(2)
                    response = chain.invoke({"question": question})
            self.add_Hisrory(response, History)
            return response
    def _call_test(self):
        import json
        with open(self.cur_question_path) as f:
            History = json.load(f)
        response = input('answer: ')
        History[-1]["ai"]=response 
        data = json.dumps(History, indent=1, ensure_ascii=False)
        with open(self.cur_question_path, 'w', newline='\n') as f:
            f.write(data)
        return response

    def add_Hisrory(self, response, History):
        History[-1]["ai"] = response.content
        import json
        data = json.dumps(History, indent=1, ensure_ascii=False)
        with open(self.cur_question_path, 'w', newline='\n') as f:
            f.write(data)
        return History

    @property
    def _llm_type(self):
        return "AnswerModel"
