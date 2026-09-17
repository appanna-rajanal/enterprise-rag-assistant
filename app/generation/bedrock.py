from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

from app.config import get_settings


def create_chat_model() -> ChatBedrock:
    """Create the AWS Bedrock chat model."""

    settings = get_settings()

    return ChatBedrock(
        model_id=settings.bedrock_chat_model_id,
        region_name=settings.aws_region,
        model_kwargs={
            "temperature": 0.1,
            "max_tokens": 1000,
        },
    )


def generate_answer(
    question: str,
    context: str,
) -> str:
    """Generate a grounded answer using retrieved context."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not context.strip():
        raise ValueError("Context cannot be empty")

    prompt = f"""
You are an enterprise network data assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."

Do not invent facts, metrics, names, or numbers.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    model = create_chat_model()

    response = model.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content
