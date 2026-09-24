from rag.retriever import KnowledgeRetriever

_retriever = None


def get_retriever():
    global _retriever

    if _retriever is None:
        _retriever = KnowledgeRetriever()

    return _retriever


def get_relevant_context(question, top_k=3):
    retriever = get_retriever()
    results = retriever.retrieve(question, top_k=top_k)

    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"Similarity: {result['score']:.4f}\n"
            f"{result['content']}"
        )

    return "\n\n".join(context_parts)