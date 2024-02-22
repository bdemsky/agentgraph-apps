import agentgraph
import os
import sys
import time
import agentgraph.config
from pathlib import Path

def writeData(scheduler, name: str, comments: str):
    nameout = name + ".comment"
    path = Path(".").absolute()
    filename = path / nameout
    filename.write_text(comments)
    
class Commenter:
    def __init__(self):
        self.model = agentgraph.LLMModel("https://demskygroupgpt4.openai.azure.com/", os.getenv("OPENAI_API_KEY"), "GPT4-8k", "GPT-32K", 34000)
        self.scheduler = agentgraph.getRootScheduler(self.model)
        self.prompts = agentgraph.Prompts("./codecomment/prompts/")
        self.sysprompt = self.prompts.loadPrompt("System")

    def process(self, names: list):
        for file in names:
            self.handleFile(file)
            
    def handleFile(self, name: str):
        path = Path(".").absolute()
        filename = path / name
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError as e:
                    pass
        else:
            return
        var = agentgraph.Var()
        self.scheduler.runLLMAgent(var, msg = self.sysprompt ** self.prompts.loadPrompt("UserPrompt", {'contents' : content}))
        self.scheduler.runPythonAgent(writeData, pos=[name, var])


def main():
    agentgraph.config.VERBOSE = 0
    commenter = Commenter()
    commenter.process(sys.argv)
    commenter.scheduler.shutdown() 

if __name__ == "__main__":
    main()
