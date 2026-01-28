import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def run_rag(retriever, question, chat_history=None):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=300
    )

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs)

    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in chat_history[-4:]
        )

    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
    You are an AI assistant designed to answer questions using ONLY the provided document context.

    Rules:
    - Use ONLY the information present in the context.
    - Do NOT use prior knowledge or make assumptions.
    - If the answer cannot be found in the context, reply exactly:
    "Answer is not available in the provided documents."
    - Be concise, factual, and accurate.
    - When relevant, quote or paraphrase directly from the context.
    """
        ),
        (
            "human",
            """
    Conversation so far:
    {history}

    Document context:
    {context}

    User question:
    {question}

    Answer:
    """
        )
    ])


    chain = prompt | llm
    response = chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

    return response.content, context
