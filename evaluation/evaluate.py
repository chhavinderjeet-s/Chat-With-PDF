from evaluation.metrics import (
    cosine_similarity_score,
    faithfulness_score
)

def evaluate_rag(answer, context, ground_truth):
    return {
        "answer_relevance_score":
            cosine_similarity_score(answer, ground_truth),
        "faithfulness_score":
            faithfulness_score(answer, context)
    }
