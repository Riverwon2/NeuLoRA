"""
ChromaDB 문서 적재 유틸리티

다양한 소스의 문서를 ChromaDB에 벡터화하여 저장합니다.
프로덕션 서비스에서 데이터를 미리 적재할 때 사용합니다.

Usage:
    # Python에서 직접 사용
    from rag.ingest import ingest_pdfs
    ingest_pdfs(["data/nlp.pdf"], persist_directory="./chroma_db")

    # CLI에서 사용
    python -m rag.ingest data/nlp.pdf data/transformer.pdf
"""

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional


def create_embedding():
    """임베딩 모델 생성 (base.py와 동일한 모델 사용)"""
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_pdfs(
    pdf_paths: List[str],
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    PDF 파일들을 ChromaDB에 적재합니다.

    Args:
        pdf_paths: PDF 파일 경로 리스트
        persist_directory: ChromaDB 저장 디렉토리
        collection_name: 컬렉션 이름
        chunk_size: 텍스트 분할 크기
        chunk_overlap: 텍스트 분할 오버랩

    Returns:
        Chroma 벡터스토어 인스턴스
    """
    # 1. 문서 로드
    docs = []
    for path in pdf_paths:
        loader = PDFPlumberLoader(path)
        loaded = loader.load()
        docs.extend(loaded)
        print(f"  📄 {path}: {len(loaded)}페이지 로드")

    print(f"📄 총 {len(docs)}개 페이지 로드 완료")

    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    print(f"✂️ 총 {len(split_docs)}개 청크로 분할 완료")

    # 3. ChromaDB에 저장
    embedding = create_embedding()
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(
        f"✅ ChromaDB 저장 완료: {persist_directory} "
        f"(collection: {collection_name}, {len(split_docs)}개 청크)"
    )

    return vectorstore


def ingest_documents(
    documents: list,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    이미 로드된 LangChain Document 리스트를 ChromaDB에 적재합니다.
    PDF 외 다른 소스(웹, DB 등)에서 가져온 문서에 활용할 수 있습니다.

    Args:
        documents: LangChain Document 리스트
        persist_directory: ChromaDB 저장 디렉토리
        collection_name: 컬렉션 이름
        chunk_size: 텍스트 분할 크기
        chunk_overlap: 텍스트 분할 오버랩

    Returns:
        Chroma 벡터스토어 인스턴스
    """
    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(documents)
    print(f"✂️ 총 {len(split_docs)}개 청크로 분할 완료")

    # ChromaDB에 저장
    embedding = create_embedding()
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(
        f"✅ ChromaDB 저장 완료: {persist_directory} "
        f"(collection: {collection_name}, {len(split_docs)}개 청크)"
    )

    return vectorstore


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingest <pdf_path1> [pdf_path2] ...")
        print("Example: python -m rag.ingest data/nlp.pdf data/transformer.pdf")
        sys.exit(1)

    ingest_pdfs(sys.argv[1:])
