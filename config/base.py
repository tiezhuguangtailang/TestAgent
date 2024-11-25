from utils.io import ShowConfigMixin


class RAGConfig(ShowConfigMixin):
    def __init__(self):
        self.model = ''
        self.tokenizer = ''

__all__ = [ "RAGConfig"]