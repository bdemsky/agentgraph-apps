import agentgraph
import itertools
import os
import sys
import time
from pathlib import Path

cur_dir = Path(os.path.dirname(os.path.realpath(__file__)))

def writeData(scheduler, name: str, comments: str):
    filename = cur_dir/ "emails" / (name + ".txt.email")
    filename.parent.mkdir(exist_ok=True, parents=True)
    filename.write_text(comments)
    
class Generator:
    def __init__(self, model):
        self.model = model
        self.scheduler = agentgraph.get_root_scheduler(self.model)
        self.prompts = agentgraph.Prompts("./emailgenerator/prompts/")
        self.sysprompt = self.prompts.load_prompt("System")
        self.syllabi = []

    def process(self, num_emails: int):
        for i, syllabus in enumerate(self.syllabi):
            for j in range(1, num_emails+1):
                self.generateEmail(syllabus, "question" + str(25 + i * num_emails + j))
       
    def loadSyllabi(self, name: str):
        filename = Path(name)
        
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                    self.syllabi.append(content)
                except UnicodeDecodeError as e:
                    
                    return
        elif filename.is_dir():
            for file in os.listdir(filename):
                self.loadSyllabi(f'{name}/{file}')

    def generateEmail(self, syllabus: str,  name: str):   
        topic = self.scheduler.run_llm_agent(msg = self.sysprompt ** self.prompts.load_prompt("PromptA", {"syllabus": syllabus}))
        email = self.scheduler.run_llm_agent(msg = self.sysprompt ** self.prompts.load_prompt("PromptB", {"syllabus": syllabus, "topic": topic}))
        self.scheduler.run_python_agent(writeData, pos=[name, email])

def main():
    agentgraph.config.DEBUG_PATH=None
    model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)

    g = Generator(model)
    file = sys.argv[1]
    num_emails = int(sys.argv[2])
    g.loadSyllabi(file)
    g.process(num_emails)
    g.scheduler.shutdown()
    

if __name__ == "__main__":
    main()
