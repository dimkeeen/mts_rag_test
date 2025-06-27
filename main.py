import os
import json
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

from hybrid_retriever import HybridRetriever, faiss_retriever, bm25_retriever

# === Логирование ===
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# === Инициализация FastAPI ===
app = FastAPI()
templates = Jinja2Templates(directory=".")

# Подключение статики (CSS и HTML из корня)
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Загрузка переменных окружения ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is required")

# === Инициализация LLM и ретривера ===
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3, max_tokens=1000)
hybrid_retriever = HybridRetriever(faiss_retriever, bm25_retriever)

# === Главная страница ===
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === Обработка запроса ===
@app.post("/ask")
async def handle_query(query: str = Form(...)):
    try:
        log.info(f"📥 Запрос: {query}")
        relevant_docs = hybrid_retriever.get_relevant_documents(query)
        log.info(f"🔍 Найдено документов: {len(relevant_docs)}")

        context = "\n".join(
            f"Вопрос: {doc.page_content}\nОтвет: {doc.metadata['answer']}"
            for doc in relevant_docs
        )

        # === Промпт через System + Human ===
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Ты ассистент по продуктам и услугам компании МТС. Отвечай только на вопросы, которые напрямую относятся к экосистеме МТС: мобильная связь, интернет, ТВ, умный дом, приложения, устройства и личный кабинет."
                " Если вопрос не связан с продуктами МТС — ответь, что ты можешь помочь только по теме МТС и не специализурешься на других вопросах."
                " Не придумывай информацию. Не отвечай на запросы вне своей области. Не пиши ссылки."
                " Отвечай строго на русском языке, в формате Markdown."
            ),
            HumanMessagePromptTemplate.from_template(
                "Вопрос: {query}\n\nКонтекст из базы знаний (если найден):\n{context}"
            )
        ])

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        answer = llm_chain.run({"context": context, "query": query})

        log.info("✅ Ответ сгенерирован")

        # Подготовка релевантных документов (без изменения ссылок)
        relevant_docs_json = [
            {
                "page_content": doc.page_content,
                "file_path": doc.metadata["file_path"]
            }
            for doc in relevant_docs
        ]

        return JSONResponse(content={
            "answer": answer.strip(),
            "relevant_docs": json.dumps(relevant_docs_json, ensure_ascii=False)
        })

    except Exception as e:
        log.error(f"❌ Ошибка: {str(e)}")
        return JSONResponse(content={
            "answer": "Произошла ошибка при обработке запроса.",
            "relevant_docs": "[]"
        })

# === Локальный запуск ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)