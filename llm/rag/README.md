# 8 Methods to Enhance the Performance of a LLM RAG Application

It is easy to prototype your first LLM RAG (Retrieval Augmented Generation) application, e.g. using this [chat-langchain][chat-langchain] template with below architecture.
![basic-rag](media/basic-rag.png)

But it is hard to make it work well. In this article, I will share some approaches to enhance the performance of the LLM RAG application.

## 1. Store message histories and user feedbacks
Chat histories and user feedbacks are important for the application analytics. We will use them later in a next session.
![rag-analysis-schema](media/rag-analysis-schema.svg)

In an above schema, 1 collection should have multiple embeddings. 1 user can have many chat sessions, and in each chat session, we store the messages (between human and AI) and their analytical information such as generated questions (condense questions or questions after query transformations), retrieved chunks and corresponding distance scores, user feedback, .etc.  

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

## 3. [Multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
When splitting documents for retrieval, there are often conflicting desires:
- We may want to have small chunks, so that their embeddings can most accurately reflect their meaning. If too long, then the embeddings can lose meaning.
- We also want to have long enough documents that the contexts are retained. Separating the document many times by `separators` and `chunk_size` sometimes breaks the context unexpectedly. It's also hard to combine the chunks in a right order to form the meaningful document for a prompt context. 

Below approaches allow us to balance precise embeddings and context retention by splitting documents into smaller chunks for embedding but retrieving larger text information or even the whole original document for the prompt context, since many LLM models nowadays support long context window, e.g. GPT-4 Turbo supports 128,000 tokens.

| <img src="media/multi-vector-retriever.png" width="50%" height="50%"> |
|:----------------------------------------------------------------------| 
| *Image by TheAiEdge.io*                                               |

- [Parent Document Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever):
  - Instead of indexing entire documents, data is divided into smaller chunks, referred to as Parent and Child documents.
  - Child documents are indexed for better representation of specific concepts, while parent documents are retrieved to ensure context retention.
  - Sometimes, the full documents can be too big to retrieve them as is. In that case, first split the raw documents into larger chunks, and then split them into smaller chunks. We then index the smaller chunks, but on retrieval we retrieve the larger chunks (but still not the full documents).
- Hypothetical Questions:
  - Documents are processed to generate potential questions they might answer. 
  - These questions are then indexed for better representation of specific concepts, while parent documents are retrieved to ensure context retention.
- Summaries:
  - Instead of indexing the entire document, a summary of the document is created and indexed. 
  - Similarly, the parent document is retrieved in the application.

## 4. Query Transformations
Because the original query can not be always optimal to retrieve for the LLM, especially in the real world. The user often doesn't provide the full context and thinks about the question from a specific angle.

Query transformation deals with transformations of the user's question before passing to the embedding model. Below are a few variations of query transform methods and their sample prompt implementation. They are all using an LLM to generate a new or multiple new queries.

1. [Rewrite-Retrieve-Read](https://github.com/langchain-ai/langchain/blob/master/cookbook/rewrite.ipynb)
2. [Hypothetical Document Embeddings (HyDE)](https://python.langchain.com/docs/templates/hyde)
3. [Follow-up question to condensed/standalone one](https://smith.langchain.com/hub/langchain-ai/weblangchain-search-query)
4. [RAG Fusion](https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb)
5. [Step-Back Prompting](https://github.com/langchain-ai/langchain/blob/master/cookbook/stepback-qa.ipynb)
6. [Multi Query Retrieval / Expansion](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)

We can also combine multiple query transformation techniques to get the best result e.g.
<img src="media/query-transformations.svg" width="200%">

## 5. Retrieval optimization
### [Self-querying](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query)
Do you remember a `metadata` column in an above `embedding` table?  We can include additional information such as author, genre, rating, the date it was written, …, and any information about the document beyond the text itself.  We can define a schema and store the metadata in a structured way alongside the vector representation.

With the database metadata schema, we use LLM to construct a structured query from the question to filter the document chunks. At the same time, the question is also converted into its vector representation for the similarity search. This kind of hybrid retrieval approaches are likely to become more and more common when RAG becomes a more widely adopted strategy.
<img src="media/self-querying.svg" width="60%">



<!-- links -->

[chat-langchain]: https://github.com/langchain-ai/chat-langchain
[ragas]: https://github.com/explodinggradients/ragas
[LangSmith]: https://smith.langchain.com
[MLflow]: https://github.com/mlflow/mlflow
[DeepEval]: https://github.com/confident-ai/deepeval
