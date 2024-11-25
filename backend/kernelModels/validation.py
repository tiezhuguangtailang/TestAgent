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
from openai import OpenAI


class Validation():
    def __init__(self) -> None:
        super().__init__()
        self.prompt_file ='PATH/TO/prompt_公积金.json'
        self.cur_question_path=''

        self.model = OpenAI(
                api_key='', base_url=""
            )
        self.model_name=""

    def prompt_with_history(self, domain='公积金'):
        import json

        prompt='''你是{domain}专家，你的任务是综合评价模型的能力。你需要从专业性和稳定性对模型的回答进行评估:
        ### 专业性评估维度：
        1. 事实正确性：提供的信息是否准确无误，与标准一致性。可以参考打分的分数。
        2. 满足用户需求：是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应。
        3. 清晰度：是否表达清晰易懂，是否使用了简洁的语言和结构，以便用户可以轻松理解。
        ### 稳定性评估维度：
        1. 一致性：模型在多次回答中是否保持信息一致性。
        2. 应对质疑能力：模型在面对质疑时是否能够坚持回答的正确性。
        3. 应对追问能力：模型在面对追问时是否能够给出进一步信息，解决评估者的追问。
        注意，回答不是越长越好，简短并且满足上述要求的回答是最好的。

        当你开始你的评估时，你需要按照遵守以下的流程：
        1. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
        2. 最后，综合每个维度的评估，为模型的专业性和稳定性分别给出一个1～10的综合分数。
        3. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
        当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
        当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
        当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
        当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
        只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。

        请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数： {{’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}}，例如：{{’一致性’: 9, ’应对质疑能力’: 6, ...,’稳定性综合得分’: 7}}，{{’事实正确性’: 9, ’满足用户需求’: 6, ...,’专业性综合得分’: 7}}
        
        以下是评估者和待测模型的对话历史和打分，评估者选择追问或质疑。
        对话历史和打分: {history}
        标准如下:{criterion}
        '''
        history = []
        criterion=[]
        json_files = [f for f in os.listdir(self.cur_question_path) if f.endswith('.json')]
        for filename in json_files:
            file_path = os.path.join(self.cur_question_path, filename)
            with open(file_path, 'r') as f:
                History = json.load(f)
                for turn in History:
                    temp = {
                        "turn": turn["turn"],
                        "问题": turn["human"],
                        "回答": turn["ai"],
                        "评价": turn["score"],
                        "分数": turn["overall_score"]
                    }
                    if turn["turn"] > 1:
                        temp['选择'] = '追问' if turn["action"] == 0 else '质疑'
                    history.append(temp)
                question=History[0]['human']
                import json
                with open(self.prompt_file) as f:
                    prompt_file = json.load(f)
                criterion.append(prompt_file[f"{question}_refine"])
        prompt=prompt.format(domain=domain, criterion=criterion, history=history)
        return prompt

    def evaluation(self, domain):
        prompt = self.prompt_with_history(domain)
        response = self.model.chat.completions.create(
              model=self.model_name,
              messages=[
                {"role": "system", "content": f"你是{domain}专家，你的任务是综合评价模型的能力。"},
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
              temperature=0
            )
      
        return response
    def add_Hisrory(self, qualitative_eval):
        import json
        with open(self.cur_question_path) as f:
            History = json.load(f)
        History.append({"qualitative_eval": qualitative_eval})
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
            print("重新打分")
            return None
    def _call(self, domain):
        score = self.evaluation(domain)
        return score
