from config.base import RAGConfig
import logging

class RAGc(RAGConfig):
    def __init__(self):

     try:
        self.embeddings_path ='PATH/TO/data/bge-m3'
        self.persist_directory = 'PATH/TO/data/retrieve_vector'
        self.bm25retriever_path ='PATH/TO/data/bm25retriever/bm25retriever.pkl'
        self.reranker_args = {'model': 'PATH/TO/bce-reranker-base_v1', 'top_n': 7, 'device': 'cuda:1'}
        self.prompt_file ='PATH/TO/backend/kernelModels/prompt_公积金.json'
        self.example_rough = 'PATH/TO/backend/kernelModels/公积金_rough.txt'
        self.result_folder = "PATH/TO/history/"
        self.initial_question='PATH/TO/backend/kernelModels/initial_question.txt'
        self.cur_question_path='PATH/TO/history/test1.json'
        self.max_turn=6
        self.alpha=1

        logging.info(f"Embeddings Path: {self.embeddings_path}")
        logging.info(f"BM25 Retriever Path: {self.bm25retriever_path}")
        logging.info(f"Prompt File: {self.prompt_file}")

        # Log success
        logging.info("RAGc initialized successfully.")


     except Exception as e:
        logging.error(f"Error initializing RAGc: {e}")
        raise