from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def cosine_similarity_score(answer, ground_truth):
    a = embeddings.embed_query(answer)
    g = embeddings.embed_query(ground_truth)
    return float(cosine_similarity([a], [g])[0][0])


def faithfulness_score(answer, context):
    grounded = sum(
        1 for sent in answer.split(".")
        if sent.strip() and sent.lower() in context.lower()
    )
    total = max(len(answer.split(".")), 1)
    return grounded / total
