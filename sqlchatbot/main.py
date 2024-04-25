import agentgraph
import argparse
import os
from pathlib import Path
import re
import sqlite3
import sys
import json

class SQLChatbot:
    def __init__(self, model, num_questions):
        self.num_questions = num_questions
        self.questions = []
        self.schema_map = {}    
        self.conn_map = {}
        self.output_dir = Path(".")
        
        self.model = model
        self.scheduler = agentgraph.get_root_scheduler(self.model)
        self.prompts = agentgraph.Prompts(os.path.dirname(os.path.realpath(__file__)) + "/prompts/")

    def loadData(self, dataset_dir, database_dir):
        tables_path = os.path.join(dataset_dir, "tables.json")
        dev_path = os.path.join(dataset_dir, "dev.json")
        self.output_dir = Path(dataset_dir)
        with open(tables_path) as inf:
            print(f"Loading tables from {tables_path}")
            tables_data= json.load(inf)
        with open(dev_path) as inf:
            print(f"Loading questions from {dev_path}")
            dev_data= json.load(inf)
        
        for db in tables_data:
            self.addDDLFromData(db)
        for db_id in self.schema_map.keys():
            self.conn_map[db_id] = sqlite3.connect(f"{database_dir}/{db_id}/{db_id}.sqlite?mode=ro", check_same_thread=False, uri=True)
        for query in dev_data:
            self.questions.append(query["question"])

    def addDDLFromData(self, db: dict):
        tables = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]
        sql_tem_all = []
        sql_tem_sim_ddl = []
        for idx, data in enumerate(tables):
            for j, column in enumerate(column_names):
                sql_tem = []
                if idx == column[0]:
                    sql_tem.append(column[0])
                    sql_tem.append(
                        str(column[1]) + " " + str(column_types[j]).upper())
                    sql_tem_all.append(sql_tem)
                    sql_tem_sim_ddl.append([column[0], column[1]])

        for foreign in db["foreign_keys"]:
            vlaus = str(tables[int(
                    column_names[foreign[0]][0])]) + "(" + str(
                column_names[foreign[0]][1]) + ") REFERENCES " + str(tables[int(
                    column_names[foreign[1]][0])]) + "(" + str(
                        column_names[foreign[1]][1]) + ")"
            sql_tem_all.append([column_names[foreign[0]][0], vlaus])
        ddl_all = []
        for idx, data in enumerate(tables):
            sql_01 = "\nCREATE TABLE " + str(data) + "("
            sql_final_tem = []
            for j, sql_final in enumerate(sql_tem_all):
                if idx == sql_final[0]:
                    sql_final_tem.append(sql_final[1])
            sql_01 += ", ".join(sql_final_tem) + ");"
            ddl_all.append(sql_01)
        self.schema_map[db["db_id"]] = ddl_all
    
    def runQueries(self):
        db_str = "\n".join(self.schema_map.keys())
        systemprompt = self.prompts.load_prompt("System")

        def getTables(_, s):
            pattern = 'database: ([a-zA-Z0-9_]+)'
            default = list(self.schema_map.keys())[0]
            matched = re.search(pattern, s)
            db = matched.group(1).lower() if matched else default
            db = db if db in self.schema_map else default
            tables = self.schema_map[db]
            return [tables, db]
        
        def getQuery(_, s):
            pattern = '```sql\n([\s\S]*)\n```'
            default = None
            matched = re.search(pattern, s)
            q = matched.group(1) if matched else default
            return [q]

        def executeQuery(_, db, query):
            conn = self.conn_map[db]
            cur = conn.cursor()
            try:
                res = cur.execute(query)
            except Exception as e:
                return [str(e)]
            all_res = str(res.fetchall())
            all_res = all_res[:20000] + "... (redacted due to length)" if len(all_res) > 20000 else all_res
            return [all_res]

        def qa2str(_, d: dict):
            return  ["\n".join([json.dumps({"question": q, "answer": a}) for q, a in d.items()])]

        qadict = agentgraph.VarDict()
        for question in self.questions[:self.num_questions]:
            var = self.scheduler.run_llm_agent(msg=systemprompt ** self.prompts.load_prompt("PromptA", {"question": question, "dbs": db_str}))
            tables, db = self.scheduler.run_python_agent(getTables, pos=[var], numOuts=2)
            var2 = self.scheduler.run_llm_agent(msg=systemprompt ** self.prompts.load_prompt("PromptB", {"question": question, "tables": tables}))
            query = self.scheduler.run_python_agent(getQuery, pos=[var2], numOuts=1)
            res = self.scheduler.run_python_agent(executeQuery, pos=[db, query], numOuts=1)
            ans = self.scheduler.run_llm_agent(msg=systemprompt ** self.prompts.load_prompt("PromptC", {"question": question, "db": db, "query": query, "res": res}))
            qadict[question] = ans

        content = self.scheduler.run_python_agent(qa2str, pos=[qadict], numOuts=1)
        self.scheduler.run_python_agent(lambda _, p, c: p.write_text(c) , pos=[self.output_dir / "dev.json.answers", content])
           
def main():
    parser = argparse.ArgumentParser(
                    prog="SQL Chatbot",
                    description="A chatbot that answers user questions by querying SQL databases"
             )
    parser.add_argument("dataset_path", 
                        help="path to the dataset")
    parser.add_argument("database_path", 
                        help="path to the database")
    args = parser.parse_args()

    model = agentgraph.LLMModel("http://127.0.0.1:8000/v1/", os.getenv("OPENAI_API_KEY"), "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", 34000, useOpenAI=True)
    # model = agentgraph.LLMModel("https://demskygroupgpt4.openai.azure.com/", os.getenv("OPENAI_API_KEY"), "GPT-35-TURBO", "GPT-35-TURBO", 34000)
    
    bot = SQLChatbot(model, 100)
    bot.loadData(args.dataset_path, args.database_path)
    bot.runQueries()
    bot.scheduler.shutdown() 

if __name__ == "__main__":
    main()
