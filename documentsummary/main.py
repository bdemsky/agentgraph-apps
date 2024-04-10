import agentgraph
import itertools
from pathlib import Path
import os
from openai import BadRequestError
import sys
import time

def writeData(scheduler, name: str, summary: str):
    dirname, fname = os.path.split(name)
    nameout = fname + ".summary"
    filename = Path(dirname) / "summary" / nameout
    filename.parent.mkdir(exist_ok=True, parents=True)
    if summary is None:
        summary = ''
    filename.write_text(summary)


class DocumentSummarizer:
    def __init__(self, model, max_len, num_docs):
        self.model = model
        self.scheduler = agentgraph.get_root_scheduler(self.model)
        self.prompts = agentgraph.Prompts("./documentsummary/prompts/")
        self.sysprompt = self.prompts.load_prompt("System")
        self.max_len = max_len
        self.num_docs = num_docs
    
    def walk_files(self, names: list):
        for name in names:
            path = Path(name)
            if path.is_dir():
                for file in os.listdir(path):
                    yield f'{name}/{file}'
            else:
                yield name

    def process(self, names: list):
        for name in itertools.islice(self.walk_files(names), self.num_docs):
            self.handleFile(name)
       
    def handleFile(self, name: str):
        filename = Path(name)
        print(filename)
        
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError as e:
                    print(e)
                    return
        else:
            return

        content = content[:self.max_len]
        var = self.scheduler.run_llm_agent(msg = self.sysprompt ** self.prompts.loadPrompt("UserPrompt", {'contents' : content}))
        self.scheduler.run_python_agent(writeData, pos=[name, var])
    
def main():
    model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)
    
    summarizer = DocumentSummarizer(model, 4000, 100)
    summarizer.process(sys.argv[1:]) # dont summarize the module itself
    summarizer.scheduler.shutdown() 

if __name__ == "__main__":
    main()
