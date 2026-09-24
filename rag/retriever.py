from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


KNOWLEDGE_DIR = (
    Path(__file__).resolve().parent / "knowledge"
)

MODEL_NAME = "all-MiniLM-L6-v2"


class KnowledgeRetriever:

    def __init__(self):

        self.chunks = []

        self._load_and_chunk_documents()

        if not self.chunks:

            raise RuntimeError(
                "No knowledge chunks found."
            )

        print(
            f"Loading embedding model: {MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        self.embeddings = self.model.encode(
            [
                chunk["content"]
                for chunk in self.chunks
            ],
            normalize_embeddings=True
        )


    def _load_and_chunk_documents(self):

        for file_path in KNOWLEDGE_DIR.glob("*.md"):

            text = file_path.read_text(
                encoding="utf-8"
            )

            sections = self._split_into_sections(
                text
            )

            for section in sections:

                if section.strip():

                    self.chunks.append(
                        {
                            "source": file_path.name,
                            "content": section.strip()
                        }
                    )


    def _split_into_sections(self, text):

        lines = text.splitlines()

        sections = []

        current_section = []

        for line in lines:

            # Start a new chunk at any ## or ### heading.
            if (
                line.startswith("## ")
                or line.startswith("### ")
            ):

                if current_section:

                    sections.append(
                        "\n".join(current_section)
                    )

                current_section = [line]

            else:

                current_section.append(line)

        if current_section:

            sections.append(
                "\n".join(current_section)
            )

        return sections


    def retrieve(
        self,
        query,
        top_k=3
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices:

            score = float(
                scores[index]
            )

            results.append(
                {
                    "source": self.chunks[index]["source"],
                    "content": self.chunks[index]["content"],
                    "score": score
                }
            )

            if len(results) >= top_k:

                break

        return results


if __name__ == "__main__":

    retriever = KnowledgeRetriever()

    print(
        f"Loaded {len(retriever.chunks)} knowledge chunks."
    )

    query = input(
        "Enter a question: "
    )

    results = retriever.retrieve(
        query
    )

    print()
    print("=" * 70)
    print("SEMANTICALLY RETRIEVED KNOWLEDGE")
    print("=" * 70)

    for result in results:

        print()

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print("-" * 70)

        print(
            result["content"]
        )