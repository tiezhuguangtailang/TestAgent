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
import time
import argparse
from typing import List

class SplitRetriever:
    def __init__(self, retriever, batch_size=6):
        self.retriever = retriever
        self.batch_size = batch_size

    def invoke(self, prompt):
        docs = self.retriever.invoke(prompt)
        first_batch = docs[:self.batch_size]
        second_batch = docs[self.batch_size:2*self.batch_size]
        third_batch = docs[2*self.batch_size:3*self.batch_size]
        forth_batch = docs[3*self.batch_size:4*self.batch_size]
        fifth_batch = docs[4*self.batch_size:5*self.batch_size]
        sixth_batch = docs[5*self.batch_size:6*self.batch_size]
        last_batch = docs[6*self.batch_size:]
        return first_batch, second_batch, third_batch, last_batch
    
class InitialQuestion(BaseLLM, LoaderMixin):
    class Config:
        extra = Extra.allow

    def __init__(self, conf) -> None:
        super().__init__()
        conf._show()
        self._load(conf)
        self.ori_model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", temperature=1.0)
        self.retriever = self.load_retriever()

    def load_retriever(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_path)

        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=self.persist_directory
        )
        chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        self.reranker_args['top_n']=10
        reranker = BCERerank(**self.reranker_args)
        compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=chroma_retriever)
        return compression_retriever
    def load_chain(self, llm, domain='公积金', verbose=False):
        template = """你是{domain}模型的用户。请你根据上下文，生成{case}场景下的不同用户问题，目的是快速探索模型的{domain}能力。确保问题以事实为基础，能够从上下文中得到答案，有逻辑性。直接列出问题，不要回答。
        可参考的上下文：
        ···
        {context}
        ···
        有用的用户问题:"""
        template=template.format(domain=domain,case='{case}',context='{context}')
        rag_prompt = PromptTemplate.from_template(template)
        self.retriever = self.load_retriever()

        rag_chain_from_docs = (rag_prompt
        | llm
        | StrOutputParser()
        )
        chain = RunnableParallel(
            { "context":  self.retriever, 
              "case": RunnablePassthrough(),
            }
        ).assign(answer=rag_chain_from_docs)

        return chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def load_model(self,domain,llm=None):
        if not llm:
            self.model = self.load_chain(self.ori_model, domain=domain)
        else:
            self.model = self.load_chain(llm, domain=domain)

    def _call(self, prompt):
        if not hasattr(self, 'model'):
            raise ValueError("Have not load model yet, please run llm.load_model() first!")
        else:
            with suppress_stdout_stderr():
                try:
                    init_Q = self.model.invoke(prompt)
                except:
                    time.sleep(2)
                    init_Q = self.model.invoke(prompt)
                print('init_Q',init_Q)
                self.store_initialQ(init_Q=init_Q['answer'], case=prompt)
            return init_Q

    def store_initialQ(self, init_Q, case):
        import re
        cleaned_questions = re.sub(r"###\d+\.\s*", "", init_Q).strip()
        cleaned_questions = re.sub(r"^\d+\.\s*", "", cleaned_questions, flags=re.MULTILINE)
        base_path = self.result_folder
        os.makedirs(base_path,exist_ok=True)
        file_name = f"initialQuestion_{case}.txt"
        self.save_new_file(base_path, file_name, cleaned_questions)

    def save_new_file(self, base_path, file_name, cleaned_questions):
        base_name, ext = os.path.splitext(file_name)
        counter = 1
        new_file_path = os.path.join(base_path, file_name)
        while os.path.exists(new_file_path):
            new_file_name = f"{base_name}_{counter}{ext}"
            new_file_path = os.path.join(base_path, new_file_name)
            counter += 1
        self.initial_question = new_file_path
        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_questions)

    @property
    def _llm_type(self):
        return "InitialQuestion"
