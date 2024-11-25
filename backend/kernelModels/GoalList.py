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
import json
import time
import argparse


class GoalList(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow
    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)
        self.ori_model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", temperature=0.7)
    def load_retriever(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_path)

        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=self.persist_directory 
        )
        chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        bm25retriever = pickle.load(open(self.bm25retriever_path, 'rb'))
        bm25retriever.k = 1
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25retriever, chroma_retriever], weights=[0.2, 0.8])
        self.reranker_args['top_n']= 6
        reranker = BCERerank(**self.reranker_args)
        compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=chroma_retriever)
        return compression_retriever

    # 输入场景下的正确和不足示例, 提出粗标准
    def rough_criterion(self, case:str = '个人公积金', llm=None, domain='公积金'):
        setattr(self, f'{domain}_rough', f'PATH/TO/{domain}_rough.txt')
        with open(getattr(self, f'{domain}_{case}_rough')) as f:
            EXAMPLE_ROUGH = f.read()

        with open(self.prompt_file) as f:
            prompt = json.load(f)

        if f"{case}_rough" in prompt:
            print(f'Warning: {case}_rough already exists in prompt.json. Return')
            return
        _temp='''
        你是{domain}专家，以下是{case}场景的一些问题、参考回答和学徒回答。\n\n{example}\n\n请一句话概括学徒回答的不足，最后，请你列出一个简短的标准列表，以便其他人可以按照这些标准对其他类似问题的回答进行评价。
        '''
        _temp = _temp.format(domain=domain,example='{example}', case="{case}")
        criterion_prompt = PromptTemplate.from_template(_temp)

        os.environ["OPENAI_API_KEY"] = ""
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.rough_criterion_chain = criterion_prompt | llm
        try:
            criterion_text = self.rough_criterion_chain.invoke(
                {"case": case, "example": EXAMPLE_ROUGH}
            )
        except:
            time.sleep(2)
            criterion_text = self.rough_criterion_chain.invoke(
            {"case": case, "example": EXAMPLE_ROUGH}
            )
        case_rough = '1. **' + criterion_text.content.split('1. **')[-1]
        prompt[case + '_rough'] = case_rough
        data = json.dumps(prompt, indent=1, ensure_ascii=False)
        with open(self.prompt_file, 'w', newline='\n') as f:
            f.write(data)
        return case_rough

    # 输入具体问题question和检索结果example_refine, 给出细化标准
    def refine_criterion(self, case, question, llm=None, domain='公积金'):
        with open(self.prompt_file) as f:
            prompt = json.load(f)
        case_rough = prompt[f'{case}_rough']
        _temp='''
        你是{domain}专家，以下是{question}问题的知识库。\n\n 知识库:\n{context}\n\n 你的任务是根据知识库细化以下回答标准，如果知识库中没有则忽略。\n\n 标准:\n{rough}\n\n 请根据知识库中包含的内容细化回答该问题相关的标准，清晰列出回答应包含的具体内容，以便测试回答的性能。不要简化内容，只列出细化后的标准，不用解释和回答。
        '''
        _temp = _temp.format(domain=domain, question='{question}',context='{context}',rough=case_rough)
        rag_prompt = PromptTemplate.from_template(_temp)

        self.retriever = self.load_retriever()
        rag_chain_from_docs = (rag_prompt
        | llm
        | StrOutputParser()
        )

        self.refine_criterion_chain = RunnableParallel(
            { "context":  self.retriever,
              "question": RunnablePassthrough(),
            }
        ).assign(answer=rag_chain_from_docs)

        with open(self.prompt_file) as f:
            prompt = json.load(f)

        if f"{question}_refine" in prompt:
            print(f'Warning: {question}_refine already exists in prompt.json, please check whether you want to overwrite it.')
            return prompt[f"{question}_refine"]

        try:
            criterion_text = self.refine_criterion_chain.invoke(question)
        except:
            time.sleep(2)
            criterion_text = self.refine_criterion_chain.invoke(question)

        prompt[f"{question}_refine"] = criterion_text['answer']
        data = json.dumps(prompt, indent=1, ensure_ascii=False)
        with open(self.prompt_file, 'w', newline='\n') as f:
            f.write(data)
        return criterion_text
        return question_refine

    def load_model(self, llm=None):
        if not llm:
            self.model = self.ori_model
        else:
            self.model = llm
    def _call(self, case, question, domain):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            # case_rough = self.rough_criterion(case, self.model, domain)
            # return case_rough
            case_refine = self.refine_criterion(case, question, self.model, domain)
            return case_refine

    @property
    def _llm_type(self):
        return "GoalList"
