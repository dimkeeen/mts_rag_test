import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from hybrid_retriever import HybridRetriever, faiss_retriever, bm25_retriever

# === 1. Загрузка переменных окружения ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is required")

# === 2. Инициализация LLM ===
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000
)

# === 3. Инициализация ретривера ===
hybrid_retriever = HybridRetriever(faiss_retriever, bm25_retriever)

# === 4. Ввод запроса от пользователя ===
query = input("\nВведите ваш вопрос: ")

# === 5. Получение релевантных документов ===
relevant_docs = hybrid_retriever.get_relevant_documents(query)

# === 6. Формирование контекста ===
context = "\n".join(
    f"Вопрос: {doc.page_content}\nОтвет: {doc.metadata['answer']}"
    for doc in relevant_docs
)

# === 7. Промпт для LLM ===
prompt_template = """
Ты онлайн ассистент по продуктам компании МТС.
Ты помогаешь пользователям, вежливо и любезно отвечая на их вопросы на основе информации из базы знаний компании.
Ответь на следующий вопрос, используя предоставленную информацию:

{context}

Вопрос: {query}
Ответ должен полным и содержательным, на основе контекста из базы знаний компании МТС.

Если релевантной информации не найдено — предложи перейти по ссылке "https://support.mts.ru/contacts".
Если вопрос не относится к продуктам МТС — объясни, что ты консультируешь только по вопросам внутри экосистемы МТС.
Будь дружелюбным и вежливым. Отвечай на русском языке.
"""

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=prompt_template
)

# === 8. Генерация ответа ===
llm_chain = LLMChain(llm=llm, prompt=prompt)
answer = llm_chain.run({"context": context, "query": query})

# === 9. Вывод результата ===
print("\n=================\n✅ Ответ:")
print(answer.strip())

print("\n🔍 Релевантные документы:")
for doc in relevant_docs:
    print(f"\n- Вопрос: {doc.page_content}\n  Ответ: {doc.metadata['answer']}\n  Ссылка: {doc.metadata['file_path']}")