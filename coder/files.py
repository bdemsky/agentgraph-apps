import re
import agentgraph
from coder.coder import Agents
from typing import List, Tuple

def parse_chat(chat: str) -> List[Tuple[str, str]]:
    """
    Extracts all code blocks from a chat and returns them
    as a list of (filename, codeblock) tuples.

    Parameters
    ----------
    chat : str
        The chat to extract code blocks from.

    Returns
    -------
    List[Tuple[str, str]]
        A list of tuples, where each tuple contains a filename and a code block.
    """
    # Get all ``` blocks and preceding filenames
    regex = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
    matches = re.finditer(regex, chat, re.DOTALL)

    files = []
    for match in matches:
        # Strip the filename of any non-allowed characters and convert / to \
        path = re.sub(r'[\:<>"|?*]', "", match.group(1))

        # Remove leading and trailing brackets
        path = re.sub(r"^\[(.*)\]$", r"\1", path)

        # Remove leading and trailing backticks
        path = re.sub(r"^`(.*)`$", r"\1", path)

        # Remove trailing ]
        path = re.sub(r"[\]\:]$", "", path)

        # Get the code
        code = match.group(2)

        # Add the file to the list
        files.append((path, code))

    # Return the files
    return files


def to_files(agents: Agents, chat: str, workspace: agentgraph.FileStore):
    """
    Parse the chat and add all extracted files to the workspace.

    Parameters
    ----------
    chat : str
        The chat to parse.
    workspace : DB
        The database containing the workspace.
    """
    files = parse_chat(chat)
    responses = list()
    for file_name, new_contents in files:
        old_contents = workspace[file_name]
        if new_contents == old_contents:
            continue;
        if old_contents == "":
            workspace[file_name] = new_contents
        else:
            varmap = agentgraph.VarMap()
            ovarA = agentgraph.Var("Response")
            syspatcher = agents.prompts.load_prompt("syspatcher.txt")
            patcher = agents.prompts.load_prompt("patcher.txt")
            agent = agentgraph.create_llm_agent(agents.model, None, ovarA, msg = syspatcher > patcher)
            agents.scheduler.add_task(agent, varmap)
            responses.append(ovarA)

    for var in responses:
        response = agents.scheduler.read_variable(var)
        for rfile, patched_contents in parse_chat(response):
            workspace[rfile] = patched_contents
