import agentgraph
import os
import sys
import time
import agentgraph.config
from pathlib import Path

def writeData(scheduler, name: str, comments: str):
    dirname, fname = os.path.split(name)
    nameout = fname + ".reply"
    filename = Path(dirname)/ "reply" / nameout
    filename.parent.mkdir(exist_ok=True, parents=True)
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
        for name in names:
            path = Path(name)
            if path.is_dir():
                for file in os.listdir(path):
                    self.handleFile(f'{name}/{file}')
            else:
                self.handleFile(name)
    
    def loadSyllabi(self, file:str):
        self.handleSyllabus(file)

    def handleSyllabus(self, name: str):
        filename = Path(name)
        # print(filename)
        
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
        filename = Path(name)
        # print(filename)
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                   
                except UnicodeDecodeError as e:
                    pass
        else:
            return
        
        var = self.scheduler.runLLMAgent(msg = self.sysprompt1 ** self.prompts.loadPrompt("UserPrompt1", {'contents' : content, 'courses':self.syllabi.keys()}))
        # print("------\n", var.getValue())
        split_at_course = course = var.getValue().strip().split('*')
        if len(split_at_course) < 2:
            print("Unable to parse course name for", name, "from LLM response")
            course = "unknown"
            syllabus = "unknown"
        else: 
            course = split_at_course[1].strip().strip("\"")
            if not course in self.syllabi:
                print("unable to find syllabus for", course, "from LLM response")
                syllabus = "unknown"
            else:
                syllabus = self.syllabi[course]
        var2 = self.scheduler.runLLMAgent(msg = self.sysprompt2 ** self.prompts.loadPrompt("UserPrompt2", {'contents' : content, 'course':course, 'syllabus': syllabus}))
        self.scheduler.runPythonAgent(writeData, pos=[name, var2])


def main():
    responder = Responder()
    responder.loadSyllabi(sys.argv[1])
    responder.process(sys.argv[2:])
    responder.scheduler.shutdown()
    

if __name__ == "__main__":
    main()
