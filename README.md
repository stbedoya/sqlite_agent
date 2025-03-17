# SQL LLM Agent 
In this series of notebooks, I develop an NBA AI agent using LLaMA 3-8B Instruct as the foundational model. The agent's primary function is to generate accurate and efficient SQL queries to extract insights from the nba_roster database. Given a structured schema and a natural language question, the agent translates the request into an executable SQL statement.

**Pipeline Overview:**

- The user submits a question in natural language.
- The model processes the question alongside relevant instructions (prompt + database schema).
- The model generates an SQL query.
- The query is executed against the database.
- The retrieved data is returned to the user.



 **I am currently building this repo** 

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/stbedoya/nba-agent.git
   cd nba-agent


## LLM Prompting and Fine-Tuning

| Description                                      | Link  |
|--------------------------------------------------|-------|
| Prompting and structured output with LLaMA 3-8B | [Notebook](https://github.com/stbedoya/sqlite_agent/blob/main/notebooks/prompting_llama3.ipynb) |

