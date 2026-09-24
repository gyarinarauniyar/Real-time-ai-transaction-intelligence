from pathlib import Path


KNOWLEDGE_DIR = (
    Path(__file__).resolve().parent / "knowledge"
)


def load_documents():

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "source": file_path.name,
                "content": text
            }
        )

    return documents


if __name__ == "__main__":

    documents = load_documents()

    print(
        f"Loaded {len(documents)} knowledge documents."
    )

    for document in documents:

        print(
            f"- {document['source']}"
        )