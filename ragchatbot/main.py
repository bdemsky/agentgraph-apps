from retriever import FAISSRetriever
import text_utils

import agentgraph
import argparse
import itertools
import os
import sys
import json
from pathlib import Path

def writeData(scheduler, name: str, content: str):
    nameout = name + ".answers"
    filename = Path(nameout)
    filename.write_text(content)

class RAGChatbot:
    def __init__(self, model, max_len, num_docs, k):
        self.max_len = max_len
        self.num_docs = num_docs
        self.k = k
        
        self.model = model
        self.scheduler = agentgraph.get_root_scheduler(self.model)
        self.prompts = agentgraph.Prompts(os.path.dirname(os.path.realpath(__file__)) + "/prompts/")
        self.systemp = self.prompts.load_prompt("System")
        self.dataset_path = None
        self.documents = []
        self.questions = []
        self.retriever = None 
    
    def loadData(self, dataset_path: str, index_path: str=None, override: bool=False, device="cpu"):
        self.dataset_path = dataset_path
        self.documents, self.questions = self._loadNQDataset(dataset_path)
        self.retriever = FAISSRetriever(index_path, override, device, self.documents)

    def _loadNQDataset(self, path) -> tuple[list[str], list[str]]:
        documents, questions = [], []
        with open(path) as f:
            for line in itertools.islice(f, self.num_docs):
                entry = text_utils.simplify_nq_example(json.loads(line))
                # save tokens by only using answer candidates
                doc = ""
                for candidate in entry["long_answer_candidates"]:
                    if candidate["top_level"] and len(doc) < self.max_len:
                        doc += entry["document_text"][candidate["start_token"]: candidate["end_token"]]
                documents.append(doc)
                questions.append(entry["question_text"])

        return documents, questions
                
    def generateAnswers(self):
        def retrieve(_, r, q, k):
            return ["\n".join(r.retrieve(q, k))]

        def qa2str(_, d: dict):
            return  ["\n".join([json.dumps({"question": q, "answer": a}) for q, a in d.items()])]

        qadict = agentgraph.VarDict()
        for question in self.questions:
            docs = self.scheduler.run_python_agent(retrieve, pos=[self.retriever, question, self.k], numOuts=1)
            answer = self.scheduler.run_llm_agent(msg=self.systemp ** self.prompts.load_prompt("PromptA", {"docs": docs, "question": question}))
            qadict[question] = answer
            
        content = self.scheduler.run_python_agent(qa2str, pos=[qadict], numOuts=1)
        self.scheduler.run_python_agent(writeData, pos=[self.dataset_path, content])


def main():
    parser = argparse.ArgumentParser(
                    prog="RAG-based Q&A bot",
                    description="A bot that is able to answer questions about user-provided documents"
             )
    parser.add_argument("dataset_path", 
                        help="path to the dataset")
    parser.add_argument("index_path",
                        default=None,
                        help="path to the faiss index")
    parser.add_argument('-r', '--override', 
                        default=False,
                        action='store_true',
                        help="whether to override existing faiss index. Only used if index exists at index_path")
    args = parser.parse_args()

    model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)
    # model = agentgraph.LLMModel("https://demskygroupgpt4.openai.azure.com/", os.getenv("OPENAI_API_KEY"), "GPT-35-TURBO", "GPT-35-TURBO", 34000)
    
    bot = RAGChatbot(model, 4000, 100, 1)
    bot.loadData(args.dataset_path, args.index_path, args.override, "cuda")
    bot.generateAnswers()
    bot.scheduler.shutdown() 

if __name__ == "__main__":
    main()
