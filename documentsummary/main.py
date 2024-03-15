import agentgraph
from pathlib import Path
import os
import sys
import time

def writeData(scheduler, name: str, summary: str):
    dirname, fname = os.path.split(name)
    nameout = fname + ".summary"
    filename = Path(dirname) / "summary" / nameout
    filename.parent.mkdir(exist_ok=True, parents=True)
    filename.write_text(summary)


class DocumentSummarizer:
    def __init__(self):
        self.model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)
        self.scheduler = agentgraph.getRootScheduler(self.model)
        self.prompts = agentgraph.Prompts("./documentsummary/prompts/")
        self.sysprompt = self.prompts.loadPrompt("System")
    
    def process(self, names: list):
        for name in names:
            path = Path(name)
            if path.is_dir():
                for file in os.listdir(path):
                    self.handleFile(f'{name}/{file}')
            else:
                self.handleFile(name)

    def handleFile(self, name: str):
        filename = Path(name)
        #print(filename)
        
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError as e:
                    print(e)
                    return
        else:
            return
        var = self.scheduler.runLLMAgent(msg = self.sysprompt ** self.prompts.loadPrompt("UserPrompt", {'contents' : content}))
        self.scheduler.runPythonAgent(writeData, pos=[name, var])
    
def main():
    summarizer = DocumentSummarizer()
    summarizer.process(sys.argv[1:]) # dont summarize the module itself
    summarizer.scheduler.shutdown() 

if __name__ == "__main__":
    main()
