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
    path = Path(".").absolute() / "bookillustrator" / "output" 
    pdfname = path / (name + ".pdf")
    pngname = path / (name + ".png")

    image.save(pngname)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(0, 10, excerpt)
    pdf.add_page()
    pdf.multi_cell(0, 10, caption)
    pdf.image(str(pngname))
    pdf.output(pdfname, 'F')
    
def genImage(scheduler, pipeline, prompt):
    image = pipeline(prompt, use_safetensors=True).images[0]
    return [image]

def genImageOpenAI(scheduler, model, prompt):
    params = {        
            "model": model.smallModel,
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "standard",
            "response_format": "b64_json",
            "n": 1,
        }
    cache_result = asyncio.run(model.lookupCache(params))
    if cache_result != None:
        image = Image.open(io.BytesIO(base64.b64decode(cache_result["data"][0]["b64_json"])))
        return [image]

    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    try:
        responseobj = asyncio.run(model.client.images.generate(**params))
    except openai.BadRequestError as e:
        if e.body["code"] == "content_policy_violation":
            print(e)
            return [None]
        else:
            raise e
    end_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    difftime = (end_time-start_time)/1000000000
    if agentgraph.config.TIMING:
        print(f"Dalle Request Time={difftime}s, prompt={prompt}, url={model.client.base_url}")

    response = responseobj.dict()
    model.writeCache(params, response)
    image = Image.open(io.BytesIO(base64.b64decode(response["data"][0]["b64_json"])))
    return [image]

def handleExcerpt(scheduler, illustrator, excerpt: str, title: str, i: int):
        image_val = None
        response_val = None

        while image_val == None:
            response = scheduler.runLLMAgent(msg = illustrator.sysprompt ** illustrator.prompts.loadPrompt("PromptA", {'excerpt': excerpt, 'title': title, "prev_response": response_val}))
            model = illustrator.image_models[i % len(illustrator.image_models)]
            image = scheduler.runPythonAgent(genImageOpenAI, pos=[model, response], numOuts=1)
            image_val = image.getValue()
            response_val = response.getValue()

        scheduler.runPythonAgent(saveAsPDF, pos=[f'{title}_{i}', image, excerpt, response])

class Illustrator:
    def __init__(self):
        self.model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)
        # self.model = agentgraph.LLMModel("https://demskygroupgpt4.openai.azure.com/", os.getenv("OPENAI_API_KEY"), "GPT-35-TURBO", "GPT-35-TURBO", 34000)
        self.scheduler = agentgraph.getRootScheduler(self.model)
        self.prompts = agentgraph.Prompts("./bookillustrator/prompts/")
        self.sysprompt = self.prompts.loadPrompt("System")
        # abuse LLMModel class for Dalle support
        self.image_models = [
                agentgraph.LLMModel("https://demskydalle.openai.azure.com/", os.getenv("OPENAI_API_KEY_DALLE"), "dalle", "dalle", 0, api_version="2024-02-01"),
                agentgraph.LLMModel("https://demskydalle2.openai.azure.com/", os.getenv("OPENAI_API_KEY_DALLE2"), "Dalle3", "Dalle3", 0, api_version="2024-02-01"),
            ]

        # self.pipeline = diffusers.DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        # self.pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config) 
        # self.pipeline.to('cuda')

        self.sep = "\n"
        self.excerpt_size = 4000
        self.num_excerpt = 1

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
        filename = path / name
        if filename.is_file():
            with open(filename, "r") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError as e:
                    pass
        else:
            return
        
        title = filename.name.split(".")[0]
        for i, excerpt in itertools.islice(enumerate(self.genExcerpts(content)), self.num_excerpt):
            self.scheduler.runPythonAgent(handleExcerpt, pos=[self, excerpt, title, i])
            
def main():
    illustrator = Illustrator()
    illustrator.process(sys.argv[1])
    illustrator.scheduler.shutdown()

if __name__ == "__main__":
    main()
