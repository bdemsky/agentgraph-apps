import agentgraph
import asyncio
import base64
import diffusers
from fpdf import FPDF
import io
import itertools
import os
import openai
from pathlib import Path
from PIL import Image
import sys
import time
import torch

def saveAsPDF(scheduler, name: str, image, excerpt: str, caption: str):
    pdfname = Path(name + ".pdf")
    pngname = Path(name + ".png")

    image.save(pngname)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(0, 10, excerpt)
    pdf.add_page()
    pdf.multi_cell(0, 10, caption)
    pdf.image(str(pngname))
    pdf.output(pdfname, 'F')
    
def genImageOpenAI(scheduler, config, prompt):
    client, model_name = config
    params = {        
            "model": model_name,
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "standard",
            "response_format": "b64_json",
            "n": 1,
        }

    try:
        responseobj = client.images.generate(**params)
    except openai.BadRequestError as e:
        if e.body["code"] == "content_policy_violation":
            print(e)
            return [None]
        raise e

    response = responseobj.dict()
    image = Image.open(io.BytesIO(base64.b64decode(response["data"][0]["b64_json"])))
    return [image]

def handleExcerpt(scheduler, illustrator, excerpt: str, file: Path, i: int):
        image_val = None
        response_val = None
        title = file.stem

        while image_val == None:
            response = scheduler.run_llm_agent(msg = illustrator.sysprompt ** illustrator.prompts.load_prompt("PromptA", {'excerpt': excerpt, 'title': title, "prev_response": response_val}))
            config = illustrator.dalle_configs[i % len(illustrator.dalle_configs)]
            image = scheduler.run_python_agent(genImageOpenAI, pos=[config, response], numOuts=1)
            image_val = image.get_value()
            response_val = response.get_value()

        scheduler.run_python_agent(saveAsPDF, pos=[str(file.parent / "output" / file.stem) + f"_{i}", image, excerpt, response])

class Illustrator:
    def __init__(self, model):
        self.model = model
        self.scheduler = agentgraph.get_root_scheduler(self.model)
        self.prompts = agentgraph.Prompts("./bookillustrator/prompts/")
        self.sysprompt = self.prompts.load_prompt("System")
        # abuse LLMModel class for Dalle support
        self.dalle_configs = [
                (openai.AzureOpenAI(azure_endpoint="https://demskydalle.openai.azure.com/", api_key=os.getenv("OPENAI_API_KEY_DALLE"), api_version="2024-02-01"), "dalle"),
                (openai.AzureOpenAI(azure_endpoint="https://demskydalle2.openai.azure.com/", api_key=os.getenv("OPENAI_API_KEY_DALLE2"), api_version="2024-02-01"), "Dalle3"),
            ]

        self.sep = "\n"
        self.excerpt_size = 4000
        self.num_excerpt = 10

    def genExcerpts(self, content: str):
        excerpt = ""
        for p in content.split(self.sep):
            if len(excerpt) + len(p) + 1 > self.excerpt_size:
                yield excerpt
                excerpt = ""
            excerpt += self.sep + p
        yield excerpt

    def process(self, name: str):
        path = Path(".").absolute()
        file = path / name
        if file.is_file():
            with open(file, "r") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError as e:
                    pass
        else:
            return
        
        for i, excerpt in itertools.islice(enumerate(self.genExcerpts(content)), self.num_excerpt):
            self.scheduler.run_python_agent(handleExcerpt, pos=[self, excerpt, file , i])
            
def main():
    model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)
    # model = agentgraph.LLMModel("https://demskygroupgpt4.openai.azure.com/", os.getenv("OPENAI_API_KEY"), "GPT-35-TURBO", "GPT-35-TURBO", 34000)
    
    illustrator = Illustrator(model)
    illustrator.process(sys.argv[1])
    illustrator.scheduler.shutdown()

if __name__ == "__main__":
    main()
