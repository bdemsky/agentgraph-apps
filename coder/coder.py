import agentgraph
import coder.files
import os
import time
import subprocess

class Agents:
    def __init__(self, problem):
        self.problem = problem
        self.filestore = agentgraph.FileStore()
        self.model = agentgraph.LLMModel("https://demskygroupgpt4.openai.azure.com/", os.getenv("OPENAI_API_KEY"), "GPT4-8k", "GPT-32K", 34000)
        self.scheduler = agentgraph.getRootScheduler(self.model)
        self.prompts = agentgraph.Prompts("./coder/prompts/")        
        
class Agent:
    def __init__(self, prompt, var):
        self.prompt = prompt
        self.conv = agentgraph.Conversation()
        self.var = var
        
        
def test():
    agents = Agents("Write a calculator")
    varmap = agentgraph.VarMap()
    agents.programmer = Agent(agents.prompts.loadPrompt("programmer.txt"), varmap.mapToNone("programmer"))
    agents.coach = Agent(agents.prompts.loadPrompt("coach.txt"), varmap.mapToNone("coach"))
    doRound(agents, agents.filestore)
    agents.scheduler.runPythonAgent(patchFiles, pos=[agents.prompts, agents.programmer.var, agents.filestore])
    agents.scheduler.runPythonAgent(compile, pos=[agents.prompts, agents.filestore, agents.coach.conv])

    agents.scheduler.shutdown()

def compile(scheduler, prompts, filestore, coachconv):
    output = filehelper(scheduler, filestore)[0]
    sysbuildprompt = prompts.loadPrompt("sysbuildeng.txt")
    buildprompt = prompts.loadPrompt("buildeng.txt", {'files': output})
    outVar = agentgraph.Var()
    scheduler.runLLMAgent(outVar, msg = sysbuildprompt ** buildprompt)
    patchFiles(scheduler, prompts, outVar.getValue(), filestore)
    filestore.writeFiles("./testdir/")
    p = subprocess.run(["bash", "compile.sh"], cwd = "./testdir/", capture_output=True)
    print(p)
    if p.returncode == 0:
        return True

    scheduler.runPythonAgent(filehelper, pos=[filestore], out=[outVar])
    scheduler.runLLMAgent(outVar, msg = prompts.loadPrompt("programmer.txt") ** ~coachconv & outVar & prompts.loadPrompt("compile.txt", {'error': p.errormsg}))
    scheduler.runPythonAgent(patchFiles, pos=[prompts, outVar, filestore])
    
    return False

    
def patchFiles(scheduler, prompts, agentOutput: str, fileStore):
    newFiles = coder.files.parse_chat(agentOutput)
    patchedFiles = dict()
    for file, newContent in newFiles:
        if file in fileStore:
            oldContent = fileStore[file]
            sysMsg = prompts.loadPrompt("syspatcher.txt")
            patchMsg = prompts.loadPrompt("patcher.txt", {'file_name': file, 'old_contents': oldContent, 'new_contents': newContent})
            patchedContents = agentgraph.Var("Patched")
            scheduler.runLLMAgent(patchedContents, msg = sysMsg ** patchMsg)
            patchedFiles[file] = patchedContents
        else:
            fileStore[file] = newContent
    for file in patchedFiles:
        fileStore[file] = patchedFiles[file].getValue()
    
def filehelper(scheduler, filestore):
    filelist = ""
    for file in filestore.get_files():
        contents = filestore[file]
        filelist+=file+"\n'''\n" + contents+"\n'''\n\n"
    return [filelist]

def doRound(agents, filestore):
    programmer = agents.programmer
    coach = agents.coach
    prompt = agents.problem
    outvar = agentgraph.Var("filecontents")
    # get files
    agents.scheduler.runPythonAgent(filehelper, pos=[agents.filestore], out=[outvar])
    
    # coach
    agents.scheduler.runLLMAgent(coach.var, conversation = coach.conv, msg = coach.conv > (coach.prompt ** agents.problem & ~coach.conv & outvar) | coach.prompt ** agents.problem)

    # programmer
    agents.scheduler.runLLMAgent(programmer.var, conversation = programmer.conv, msg = programmer.conv > (programmer.prompt ** ~coach.conv[:-1] & outvar & coach.var | programmer.prompt ** coach.var))

    
if __name__ == "__main__":
    test()
