"""Module containing the system message for the Spark SQL agent."""

system_prompt = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SparkSQL query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {tables}

If you need to filter on a proper company name, you must ALWAYS first look up the filter value using the "search_proper_company_name" tool!
Do not try to guess at the proper name - use this function to find similar ones."""

search_proper_nouns_tool_name = "search_proper_company_name"
search_proper_nouns_tool_description = """Use to look up values to filter on company name. \
Input is an approximate spelling of the company name, output is valid proper company name. \
Use the name most similar to the search."""

more_step_message_content = "Sorry, need more steps to process this request."

few_shot_header = (
    "\n\nHere are some examples of user inputs and their corresponding SQL queries:\n"
)
few_shot_examples = [
    {
        "input": "List all the project_name and amount from the opportunities table",
        "query": "SELECT project_name, amount\nFROM opportunities",
    },
    {
        "input": "Count the total number of leads in the leads table",
        "query": "SELECT COUNT(*) AS total_leads\nFROM leads",
    },
    {
        "input": "Retrieve all unique lead_status from the leads table",
        "query": "SELECT DISTINCT lead_status\nFROM leads",
    },
    {
        "input": "Find all company_name and state from the accounts table where the city is Miami",
        "query": "SELECT company_name, state\nFROM accounts\nWHERE city = 'Miami'",
    },
    {
        "input": "Calculate the total sum of amount for opportunities in the Closed Won stage",
        "query": "SELECT SUM(amount) AS total_won_amount\nFROM opportunities\nWHERE stage = 'Closed Won'",
    },
    {
        "input": "Join opportunities and accounts tables to list all opportunities with their corresponding company_name",
        "query": "SELECT o.oppurtunity_name, a.company_name, o.amount\nFROM opportunities o\nJOIN accounts a ON o.company_ext_id = a.company_ext_id",
    },
    {
        "input": "Find the total number of opportunities by each currency type",
        "query": "SELECT currency, COUNT(*) AS opportunity_count\nFROM opportunities\nGROUP BY currency",
    },
    {
        "input": "Use a Common Table Expression (CTE) to find the opportunities with amounts greater than the average amount for all opportunities",
        "query": """WITH avg_amount_cte AS (
    SELECT AVG(amount) AS avg_amount
    FROM opportunities
)
SELECT oppurtunity_name, amount
FROM opportunities, avg_amount_cte
WHERE amount > avg_amount_cte.avg_amount""",
    },
    {
        "input": "Use a window function to calculate the running total of amount for opportunities in the Closed Won stage, ordered by close_date",
        "query": """SELECT oppurtunity_name, close_date, amount,
       SUM(amount) OVER (ORDER BY close_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM opportunities
WHERE stage = 'Closed Won'""",
    },
    {
        "input": "Perform a multi-table join to retrieve the first_name, last_name, company_name, and oppurtunity_name for contacts related to opportunities that are in the Closed Lost stage",
        "query": """SELECT c.first_name, c.last_name, a.company_name, o.oppurtunity_name
FROM contacts c
JOIN accounts a ON c.company_ext_id = a.company_ext_id
JOIN opportunities o ON a.company_ext_id = o.company_ext_id
WHERE o.stage = 'Closed Lost'""",
    },
]
