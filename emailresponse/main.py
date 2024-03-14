import agentgraph
import os
import sys
import time
import agentgraph.config
from pathlib import Path

def writeData(scheduler, name: str, comments: str):
    nameout = name + ".reply"
    path = Path(".").absolute()
    filename = path / nameout
    filename.write_text(comments)
    
class Responder:
    def __init__(self):
        self.model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)
        self.scheduler = agentgraph.getRootScheduler(self.model)
        self.prompts = agentgraph.Prompts("./emailresponse/prompts/")
        self.sysprompt1 = self.prompts.loadPrompt("System1")
        self.sysprompt2 = self.prompts.loadPrompt("System2")
        self.syllabi = dict()
    def process(self, names: list):
        for file in names:
            self.handleFile(file)
    
    def loadSyllabi(self, file:str):
        self.handleSyllabus(file)

    def handleSyllabus(self, name: str):
        path = Path(".").absolute()
        filename = path / name
        print(filename)
        
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                    self.syllabi[Path(name).stem]=content
                except UnicodeDecodeError as e:
                    
                    return
        elif filename.is_dir():
            for file in os.listdir(filename):
                self.handleSyllabus(f'{name}/{file}')
            
        return


    def handleFile(self, name: str):
        path = Path(".").absolute()
        filename = path / name
        print(filename)
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError as e:
                    pass
        else:
            return
        var = self.scheduler.runLLMAgent(msg = self.sysprompt1 ** self.prompts.loadPrompt("UserPrompt1", {'contents' : content, 'courses':self.syllabi.keys()}))
        print("------\n", var.getValue())
        course = var.getValue().strip().split('*')[1].strip().strip("\"")
        var2 = self.scheduler.runLLMAgent(msg = self.sysprompt2 ** self.prompts.loadPrompt("UserPrompt2", {'contents' : content, 'course':course, 'syllabus': self.syllabi[course]}))
        self.scheduler.runPythonAgent(writeData, pos=[name, var2])


def main():
    agentgraph.config.VERBOSE = 1
    responder = Responder()
    responder.loadSyllabi(sys.argv[1])
    responder.process(sys.argv[2:])
    responder.scheduler.shutdown()
    

if __name__ == "__main__":
    main()