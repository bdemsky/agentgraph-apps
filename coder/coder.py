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
        self.scheduler = agentgraph.get_root_scheduler(self.model)
        self.prompts = agentgraph.Prompts("./coder/prompts/")        
        
class Agent:
    def __init__(self, prompt, var):
        self.prompt = prompt
        self.conv = agentgraph.Conversation()
        self.var = var
        
        
def test():
    agents = Agents("a command line based full featured scientic calculator in C that supports standard feature including trig functions and parenthesis. The calculator should take the problem instance from the command line. You should compile the program such that it can be executed by typing ./calculator.  For example, ./calculator \"sin(90) + 1\" should print 2. \n ./calculator \"3 * (2 + 4)\" should print 18. \n ./calculator \"1/2\" should print 0.5. ./calculator \"1.5 * 2.0\" should print 3.  ./calculator \"2*(1+3*(2-1))\" should print 8.\n")
    varmap = agentgraph.VarMap()
    agents.programmer = Agent(agents.prompts.load_prompt("sysprogrammer.txt"), varmap.map_to_none("programmer"))
    agents.coach = Agent(agents.prompts.load_prompt("syscoach.txt"), varmap.map_to_none("coach"))
    mainloop(agents)

def buildTests(agents: Agents):
    scheduler = agents.scheduler
    sys = agents.prompts.load_prompt("testeng.txt")
    messages = agents.prompts.load_prompt("testprompt.txt", { 'agentproblem' : agents.problem})
    outVar = scheduler.run_llm_agent(msg = sys ** messages)
    scheduler.run_python_agent(patchFiles, pos=[agents.prompts, outVar, agents.filestore])

def doTests(agents: Agents):
    agents.filestore.write_files("./testdir/")
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
        agents.filestore.wait_for_access()
        # Prompt user for input
        message = input("(Q)uit, (C)ompile, (P)rogram, c(R)eate Tests, run (T)est, (D)ump code, (M)anual request: ")
            # Exit program if user inputs "quit"
        if message.lower() == "q":
            break
        elif message.lower() == "c":
            agents.scheduler.run_python_agent(compile, pos=[agents.prompts, agents.filestore, agents.coach.conv])
        elif message.lower() == "r":
            buildTests(agents)
        elif message.lower() == "p":
            doCoachProgram(agents)
        elif message.lower() == "t":
            doTests(agents)
        elif message.lower() == "d":
            dump(agents.filestore)
        elif message.lower() == "m":
            manual(agents)

    agents.scheduler.shutdown()

def dump(filestore):
    filestore.write_files("./testdir/")
    
def compile(scheduler, prompts, filestore, coachconv):
    output = filehelper(scheduler, filestore)[0]
    sysbuildprompt = prompts.load_prompt("sysbuildeng.txt")
    buildprompt = prompts.load_prompt("buildeng.txt", {'files': output})
    outVar = scheduler.run_llm_agent(msg = sysbuildprompt ** buildprompt)
    patchFiles(scheduler, prompts, outVar.get_value(), filestore)
    dump(filestore)
    p = subprocess.run(["bash", "compile.sh"], cwd = "./testdir/", capture_output=True)
    print(p)
    if p.returncode == 0:
        return True

    outVar = scheduler.run_python_agent(filehelper, pos=[filestore], numOuts = 1)
    outVar = scheduler.run_llm_agent(msg = prompts.load_prompt("sysprogrammer.txt") ** ~coachconv & outVar & prompts.load_prompt("compile.txt", {'error': p.stderr.decode()}))
    scheduler.run_python_agent(patchFiles, pos=[prompts, outVar, filestore])
    
    return False

    
def patchFiles(scheduler, prompts, agentOutput: str, fileStore):
    newFiles = coder.files.parse_chat(agentOutput)
    patchedFiles = dict()
    for file, newContent in newFiles:
        print(file, newContent)
        if file in fileStore:
            oldContent = fileStore[file]
            if oldContent != newContent:
                sysMsg = prompts.load_prompt("syspatcher.txt")
                patchMsg = prompts.load_prompt("patcher.txt", {'file_name': file, 'old_contents': oldContent, 'new_contents': newContent})
                patchedContents = agentgraph.Var("Patched")
                scheduler.run_llm_agent(patchedContents, msg = sysMsg ** patchMsg)
                patchedFiles[file] = patchedContents
        else:
            fileStore[file] = newContent
    for file in patchedFiles:
        pcontent = patchedFiles[file].get_value()
        pFiles = coder.files.parse_chat(pcontent)
        for pfile, pcontent in pFiles:
            fileStore[pfile] = pcontent
    
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
    # get files contents
    fsvar = agents.scheduler.run_python_agent(filehelper, pos=[agents.filestore])
    
    # query programmer to fix
    programmer.var = agents.scheduler.run_llm_agent(msg = programmer.prompt ** ~coach.conv & fsvar + errorvar)

    # Update filestore
    agents.scheduler.run_python_agent(patchFiles, pos=[agents.prompts, agents.programmer.var, agents.filestore])


def doCoachProgram(agents):
    filestore = agents.filestore
    programmer = agents.programmer
    coach = agents.coach
    prompt = agents.prompts.load_prompt("coach.txt", {'problem':agents.problem})
    # get files
    outvar = agents.scheduler.run_python_agent(filehelper, pos=[agents.filestore], out=[agentgraph.VarType])
    
    # coach
    coach.var = agents.scheduler.run_llm_agent(conversation = coach.conv, msg = coach.conv > (coach.prompt ** prompt & ~coach.conv & outvar) | coach.prompt ** prompt)

    # programmer
    programmer.var = agents.scheduler.run_llm_agent(conversation = programmer.conv, msg = programmer.conv > (programmer.prompt ** ~coach.conv[:-1] & outvar & coach.var | programmer.prompt ** coach.var))

    print(programmer.var.get_value())

    # Update filestore
    agents.scheduler.run_python_agent(patchFiles, pos=[agents.prompts, agents.programmer.var, agents.filestore])

if __name__ == "__main__":
    test()
