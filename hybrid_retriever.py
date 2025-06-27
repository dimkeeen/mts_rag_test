import os
import pandas as pd
import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# === Логирование ===
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CSV_FILE = "mts_questions_answers.csv"
FAISS_INDEX_DIR = "faiss_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LOCAL_MODEL_PATH = "models/paraphrase-multilingual-mpnet-base-v2"

# === 1. Загрузка и подготовка данных ===
log.info("📄 Загрузка и подготовка данных из CSV...")
df = pd.read_csv(CSV_FILE).fillna("")
documents = [
    Document(
        page_content=row["question_text"],
        metadata={
            "answer": row["answer_text"],
            "file_path": row["file_path"]
        }
    )
    for _, row in df.iterrows()
]
log.info(f"✅ Загружено документов: {len(documents)}")

# === 2. Embedding модель ===
if not os.path.exists(LOCAL_MODEL_PATH):
    log.info(f"⬇️ Модель не найдена локально, загружаем и сохраняем в {LOCAL_MODEL_PATH}...")
    model = SentenceTransformer(MODEL_NAME)
    model.save(LOCAL_MODEL_PATH)
    log.info("✅ Модель успешно загружена и сохранена")
else:
    log.info(f"📁 Используется локальная модель из {LOCAL_MODEL_PATH}")

log.info(f"🧠 Загрузка модели эмбеддингов из {LOCAL_MODEL_PATH}...")
embedding_model = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)
log.info("✅ Модель эмбеддингов загружена")

# === 3. FAISS индекс ===
if not os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
    log.info("🆕 FAISS индекс не найден, создаём новый...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(FAISS_INDEX_DIR)
    log.info("✅ FAISS индекс создан и сохранён")
else:
    log.info("📦 Загрузка существующего FAISS индекса...")
    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
    log.info("✅ FAISS индекс загружен")

faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === 4. BM25 Retriever ===
log.info("📚 Инициализация BM25 ретривера...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3
log.info("✅ BM25 ретривер инициализирован")

# === 5. Гибридный поиск ===
class HybridRetriever:
    def __init__(self, faiss_retriever, bm25_retriever):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever

    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        log.info(f"\n🔍 Выполняется гибридный поиск для запроса: {query}")
        faiss_docs = self.faiss_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        log.info(f"📎 FAISS вернул {len(faiss_docs)} документов")
        log.info(f"📎 BM25 вернул {len(bm25_docs)} документов")

        # Удаляем дубликаты
        seen = set()
        unique_docs = []
        for doc in faiss_docs + bm25_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        log.info(f"🔗 После удаления дубликатов: {len(unique_docs)} уникальных документов")
        return unique_docs

# === 6. Пример использования гибридного ретривера ===
if __name__ == "__main__":
    log.info("\n🚀 Тестирование гибридного ретривера...")
    hybrid_retriever = HybridRetriever(faiss_retriever, bm25_retriever)

    query = "как пополнить баланс мобильного?"
    relevant_docs = hybrid_retriever.get_relevant_documents(query)

    log.info("\n📌 Результаты гибридного поиска:")
    for doc in relevant_docs:
        print(f"\n📌 Вопрос: {doc.page_content}\n📝 Ответ: {doc.metadata['answer']}\n🔗 Ссылка: {doc.metadata['file_path']}")
        print("-" * 40)