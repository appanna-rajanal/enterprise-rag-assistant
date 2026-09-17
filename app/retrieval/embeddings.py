from langchain_aws import BedrockEmbeddings

from app.config import get_settings


def create_embeddings() -> BedrockEmbeddings:
    """Create the AWS Bedrock embedding model."""

    settings = get_settings()

    return BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
    )
