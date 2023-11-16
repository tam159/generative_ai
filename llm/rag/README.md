# 8 Methods to Enhance the Performance of a LLM RAG Application

It is easy to prototype your first LLM RAG (Retrieval Augmented Generation) application, e.g. using this [chat-langchain][chat-langchain] template with below architecture.
![basic-rag](media/basic-rag.png)

But it is hard to make it work well. In this article, I will share some approaches to enhance the performance of the LLM RAG application.

## 1. Store message histories and user feedbacks
Chat histories and user feedbacks are important for the application analytics. We will use them later in a next session.
![rag-analysis-schema](media/rag-analysis-schema.svg)

In an above schema, 1 collection should have multiple embeddings. 1 user can have many chat sessions, and in each chat session, we store the messages (between human and AI) and their analytical information such as generated questions (condense questions or questions after query transformations), retrieved chunks and corresponding distance scores, user feedback, .etc  

## 2. Start evaluating your application
A naive RAG app can have some challenges, e.g.: bad retrieval (low precision, low recall, outdated information), bad response generation (hallucination, irrelevance, toxicity/bias), .etc

Before improving it, we need a way to measure its performance. We can use the following [ragas][ragas] metrics for the evaluation.
![ragas-scores](media/ragas-scores.svg)

Because some scores need `ground_truth`, there are 2 ways to create the labeled evaluation dataset:
- Human-annotated labeled dataset: we can use some real questions and answers that users give good feedback from the first method `chat_analysis` table.
- Generated labeled dataset (good for cold start):
  - Prompt GPT-4 Turbo to generate questions from each chunk or multiple chunks to get pairs of question & doc chunks.
  - Run above pairs of question & context through GPT-4 Turbo to generate answer.

Eventually we can run through this labeled dataset with above pre-defined metrics and use GPT-4 Turbo as an evaluator/judge.

For a better observation, we should also use some LLMOps platforms such as [LangSmith][LangSmith], [MLflow][MLflow] or integrate [DeepEval][DeepEval] in CI/CD pipelines.


<!-- links -->

[chat-langchain]: https://github.com/langchain-ai/chat-langchain
[ragas]: https://github.com/explodinggradients/ragas
[LangSmith]: https://smith.langchain.com
[MLflow]: https://github.com/mlflow/mlflow
[DeepEval]: https://github.com/confident-ai/deepeval
