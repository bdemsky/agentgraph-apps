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
    agents = Agents("a command line based full featured scientic calculator in C that supports standard feature including trig functions and parenthesis. The calculator should take the problem instance from the command line. You should compile the program such that it can be executed by typing ./calculator.  For example, ./calculator \"sin(90) + 1\" should print 2. \n ./calculator \"3 * (2 + 4)\" should print 18. \n ./calculator \"1/2\" should print 0.5. ./calculator \"1.5 * 2.0\" should print 3.  ./calculator \"2*(1+3*(2-1))\" should print 8.\n")
    varmap = agentgraph.VarMap()
    agents.programmer = Agent(agents.prompts.loadPrompt("programmer.txt"), varmap.mapToNone("programmer"))
    agents.coach = Agent(agents.prompts.loadPrompt("coach.txt"), varmap.mapToNone("coach"))
    mainloop(agents)

def buildTests(agents: Agents):
    scheduler = agents.scheduler
    sys = agents.prompts.loadPrompt("testeng.txt")
    messages = agents.prompts.loadPrompt("testprompt.txt", { 'agentproblem' : agents.problem})
    outVar = agentgraph.Var()
    scheduler.runLLMAgent(outVar, msg = sys ** messages)
    scheduler.runPythonAgent(patchFiles, pos=[agents.prompts, outVar, agents.filestore])

def doTests(agents: Agents):
    agents.filestore.writeFiles("./testdir/")
    testout, passed = runTests(agents)
    if passed == True:
        print("Test passed")
        return
    print("Test failed")
    doProgram(agents, testout);
    
def runTests(agent:Agents):
    exception = False
    p = None
    try:
        p = subprocess.run(["bash", "test.sh"], cwd="./testdir/", timeout=5, capture_output=True)
    except subprocess.TimeoutExpired:
        print("Timeout")
        if p is None:
            msgs = "The test failed due to a timeout.  There was no output."
        else:
            msgs = "The test failed due to a timeout.  The test outputted:\n" + p.stdout.decode() + p.stderr.decode()
        exception = True
      
    if (exception==False and p.returncode==0):
        print("Test passed!\n")
        return "The output of running the test script was:\n"+p.stdout.decode()+p.stderr.decode(), True
    
    if not exception:
        msgs = "Test failed with the following message:\n" + p.stdout.decode() + p.stderr.decode()

    msgs += "If you believe this is due to an error in the program, fix the appropriate program files.\n  If you believe this is due to an error in the test setup, please output a corrected testing script file called test.sh.  Do not drop test cases from test.sh.\n"

    print("Test failed:"+msgs)
    return msgs, False
   
def mainloop(agents):
    # Keep repeating the following
    while True:
        # Prompt user for input
        message = input("(Q)uit, (C)ompile, (P)rogram, c(R)eate Tests, run (T)est, (D)ump code, (M)anual request: ")
            # Exit program if user inputs "quit"
        if message.lower() == "q":
            break
        elif message.lower() == "c":
            agents.scheduler.runPythonAgent(compile, pos=[agents.prompts, agents.filestore, agents.coach.conv])
        elif message.lower() == "r":
            buildTests(agents)
        elif message.lower() == "p":
            doCoachProgram(agents)
        elif message.lower() == "t":
            doTests(agents)
        elif message.lower() == "d":
            dump(agents)
        elif message.lower() == "m":
            manual(agents)





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
    scheduler.runLLMAgent(outVar, msg = prompts.loadPrompt("programmer.txt") ** ~coachconv & outVar & prompts.loadPrompt("compile.txt", {'error': p.stderr.decode()}))
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

def doProgram(agents, errorvar):
    filestore = agents.filestore
    programmer = agents.programmer
    coach = agents.coach
    prompt = agents.problem
    fsvar = agentgraph.Var()
    # get files contents
    agents.scheduler.runPythonAgent(filehelper, pos=[agents.filestore], out=[fsvar])
    
    # query programmer to fix
    agents.scheduler.runLLMAgent(programmer.var, msg = programmer.prompt ** ~coach.conv & fsvar + errorvar)
    # Update filestore
    agents.scheduler.runPythonAgent(patchFiles, pos=[agents.prompts, agents.programmer.var, agents.filestore])


def doCoachProgram(agents):
    filestore = agents.filestore
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

    # Update filestore
    agents.scheduler.runPythonAgent(patchFiles, pos=[agents.prompts, agents.programmer.var, agents.filestore])

if __name__ == "__main__":
    test()
