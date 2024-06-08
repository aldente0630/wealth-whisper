from datetime import datetime
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pytz import timezone

MULTI_PROMPT_ROUTER_SYSTEM_TEMPLATE = """
Given a raw text input to a language model select the model prompt best suited for the input. You will be given
the names of the available prompts and a description of what the prompt is best suited for. You may also revise 
the original input if you think that revising it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT"
if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< CHAT HISTORY >>
{{chat_history}}
"""

MULTI_PROMPT_ROUTER_HUMAN_TEMPLATE = """
<< INPUT >>
{question}
"""

MULTI_PROMPT_ROUTER_AI_TEMPLATE = """
<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>"""

CONDENSE_QUESTION_SYSTEM_PROMPT_TEMPLATE = """
Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question, in its original language.

However, if you think chat history and a follow up input question are unrelated,
ignore chat history and create a standard standalone question using only the follow up input question. 
Think step by step.

Chat History:
{chat_history}
"""

CONDENSE_QUESTION_HUMAN_PROMPT_TEMPLATE = "Follow Up Input: {question}"

CONDENSE_QUESTION_AI_PROMPT_TEMPLATE = "Standalone Question:"

QA_SYSTEM_PROMPT_TEMPLATE = """
You are an AI investment analyst assistant that uses analyst reports to answer client questions. Please adhere to 
the following guidelines when responding:

1. Base your answers on the context provided in the analyst reports. If the necessary information is unavailable or 
you are unsure, state that you lack sufficient information for a complete response.
2. Respond in the language the question was asked in to ensure effective communication.
3. Use Markdown syntax to format your responses, enhancing readability and structure.
4. Today's date is {curr_date}. For questions about current events, recent developments, or future predictions, answer 
as if it is {curr_date}, even if your knowledge stops at an earlier date. Do not include information 
after your knowledge cutoff, but reason about the future from the {curr_date} perspective. Express uncertainty 
for information between your knowledge cutoff and {curr_date}.
5. To protect privacy, do not disclose the names of individual investment firms or analysts mentioned 
in the financial reports.
6. Refrain from answering questions about real-time financial asset prices (such as stock prices), 
as the knowledge base does not contain this information.
7. When referring to charts, figures, or other visual elements from the reports, describe their content and insights 
without directing the client to view them, as you cannot display these visuals directly.
8. When you have past and current reports on a topic, prioritize the most recent and reliable information.
9. If reports conflict, carefully assess each perspective's rationale and credibility. Choose the most reasonable and 
well-supported information for your answer.
10. Avoid making definitive forecasts about the future performance of financial assets, markets, or economies. 
Instead, offer insights based on historical data, observed trends, and underlying fundamentals.
11. Do not provide personal financial advice or product recommendations. Emphasize that each person's situation is 
unique and requires tailored professional guidance.

Address each question step-by-step, breaking down complex issues. Provide clear, concise, and well-reasoned answers 
that demonstrate your expertise and professionalism. Inform clients of any limitations on answer specificity 
based on the available information.

{{context}}
"""

QA_HUMAN_PROMPT_TEMPLATE = "Question: {question}"

QA_AI_PROMPT_TEMPLATE = "Helpful Answer:"

QUESTIONS_CREATION_SYSTEM_PROMPT_TEMPLATE = """
Here's a specific topic and the context for it.

Topic:
{heading}

Context:
{body}

You generate questions based on the instructions below with a given title and context instead of prior knowledge.

You are a professor. Your job as a professor is to provide {n_questions_per_chunk} questions 
for an upcoming quiz/exam.

- Questions should be answerable with the context provided. 
- Questions should not contain complex assumptions, reasoning, or open discussion possibilities.
- Questions should be varied across the context provided. 
- Each question must begin with -.
- Write in Korean.
"""

QUESTIONS_CREATION_AI_PROMPT_TEMPLATE = "Question:"

ANSWERS_CREATION_SYSTEM_PROMPT_TEMPLATE = """
Here's the context for a specific topic. 

Context:
{body}

Using only the context provided, answer the following questions using the rules below.

- Answer questions only if you can answer them with certainty from the context given.
- If you can't answer with the context given, say, "I can't find the answer in the context given."
- Skip the preamble and get right to the point.
- Make sure your answer is long enough and detailed enough.    
- Be polite and courteous in your responses.
- Write in Korean.
"""

ANSWERS_CREATION_HUMAN_PROMPT_TEMPLATE = "Question: {question}"

HYDE_SYSTEM_PROMPT_TEMPLATE = """
Please write a concise passage that answers the question.
"""

HYDE_HUMAN_PROMPT_TEMPLATE = "Question: {query}"

HYDE_AI_PROMPT_TEMPLATE = "Passage:"

ANSWERS_CREATION_AI_PROMPT_TEMPLATE = "Answer:"

RAG_FUSION_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: {query}
"""

RAG_FUSION_AI_PROMPT_TEMPLATE = "OUTPUT ({query_augmentation_size} queries):"

SUMMARY_SYSTEM_PROMPT_TEMPLATE = """
Progressively summarize the lines of conversation provided, adding onto the previous summary 
returning a new summary.

EXAMPLE
Current summary:
The user asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is 
a force for good.

New lines of conversation:
User: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help users reach their full potential.

New summary:
The user asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is 
a force for good because it will help users reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}
"""

SUMMARY_AI_PROMPT_TEMPLATE = "New summary:"


def get_multi_prompt_router_prompt(destinations: str) -> ChatPromptTemplate:
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                MULTI_PROMPT_ROUTER_SYSTEM_TEMPLATE.format(destinations=destinations),
                input_variables=["chat_history"],
            ),
            HumanMessagePromptTemplate.from_template(
                MULTI_PROMPT_ROUTER_HUMAN_TEMPLATE,
                input_variables=["question"],
            ),
            AIMessagePromptTemplate.from_template(
                MULTI_PROMPT_ROUTER_AI_TEMPLATE,
            ),
        ]
    )
    chat_prompt_template.output_parser = RouterOutputParser()
    return chat_prompt_template


def get_answers_creation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                ANSWERS_CREATION_SYSTEM_PROMPT_TEMPLATE,
                input_variables=["body"],
            ),
            HumanMessagePromptTemplate.from_template(
                ANSWERS_CREATION_HUMAN_PROMPT_TEMPLATE,
                input_variables=["question"],
            ),
            AIMessagePromptTemplate.from_template(ANSWERS_CREATION_AI_PROMPT_TEMPLATE),
        ]
    )


def get_condensed_question_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                CONDENSE_QUESTION_SYSTEM_PROMPT_TEMPLATE,
                input_variables=["chat_history"],
            ),
            HumanMessagePromptTemplate.from_template(
                CONDENSE_QUESTION_HUMAN_PROMPT_TEMPLATE,
                input_variables=["question"],
            ),
            AIMessagePromptTemplate.from_template(CONDENSE_QUESTION_AI_PROMPT_TEMPLATE),
        ]
    )


def get_hyde_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                HYDE_SYSTEM_PROMPT_TEMPLATE, input_variables=["context"]
            ),
            HumanMessagePromptTemplate.from_template(
                HYDE_HUMAN_PROMPT_TEMPLATE, input_variables=["question"]
            ),
            AIMessagePromptTemplate.from_template(HYDE_AI_PROMPT_TEMPLATE),
        ]
    )


def get_qa_prompt() -> ChatPromptTemplate:
    curr_date = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d")
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                QA_SYSTEM_PROMPT_TEMPLATE.format(curr_date=curr_date),
                input_variables=["context"],
            ),
            HumanMessagePromptTemplate.from_template(
                QA_HUMAN_PROMPT_TEMPLATE, input_variables=["question"]
            ),
            AIMessagePromptTemplate.from_template(QA_AI_PROMPT_TEMPLATE),
        ]
    )


def get_questions_creation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                QUESTIONS_CREATION_SYSTEM_PROMPT_TEMPLATE,
                input_variables=["heading", "body", "n_questions_per_chunk"],
            ),
            AIMessagePromptTemplate.from_template(
                QUESTIONS_CREATION_AI_PROMPT_TEMPLATE
            ),
        ]
    )


def get_rag_fusion_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                RAG_FUSION_SYSTEM_PROMPT_TEMPLATE,
                input_variables=["query"],
            ),
            AIMessagePromptTemplate.from_template(
                RAG_FUSION_AI_PROMPT_TEMPLATE,
                input_variables=["query_augmentation_size"],
            ),
        ]
    )


def get_summary_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                SUMMARY_SYSTEM_PROMPT_TEMPLATE,
                input_variables=["summary", "new_lines"],
            ),
            AIMessagePromptTemplate.from_template(SUMMARY_AI_PROMPT_TEMPLATE),
        ]
    )
