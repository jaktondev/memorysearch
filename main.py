# Imports
import os
import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchResults
from bs4 import BeautifulSoup
import requests
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from json import loads
from pathlib import Path
from rich import print
from rich.progress import Progress
from rich.prompt import Prompt

# Obtener API de OpenAI desde variables de entorno
OPENAI_API = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada.")
DBPATH = "./chroma_db"
IDEAPATH = "./ideas"
TEXTS_DIR = "./texts"

# Clase TypedDict para el estado al usar RAG
class RAGState(TypedDict):
    "Estados para el flujo RAG."
    messages: Annotated[List[BaseMessage], operator.add]
    context: str

# LLMS principales
boss = ChatOpenAI(model="gpt-5", temperature=0.2, api_key=OPENAI_API, reasoning_effort="low", verbosity="low")
joiner = ChatOpenAI(model="gpt-5", temperature=0.2, api_key=OPENAI_API, reasoning_effort="medium")

# Inicializar ChromaDB, una de investigaciones echas anteriormente por el agente y otra de contexto obtenido de la web anterior
chromaclient = chromadb.PersistentClient(path=DBPATH)
chromacollection = chromaclient.get_or_create_collection(name="documents", embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=OPENAI_API))
chromaIclient = chromadb.PersistentClient(path=IDEAPATH)
chromaIcollection = chromaIclient.get_or_create_collection(name="ideas", embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=OPENAI_API))

# Agregar documentos a la base de datos de información web
def add_chroma_documents(texts: List[str]):
    """Agrega documentos a ChromaDB."""
    
    count = len(list(Path(TEXTS_DIR).glob("*.txt")))
    
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=250, 
        chunk_overlap=50, 
        length_function=len,
        is_separator_regex=False,
    )
    
    # Hacer en chunks para mejor busqueda
    for i, text in enumerate(texts):

        doc_id_prefix = f"doc_{count + i}"
        with open(Path(TEXTS_DIR) / f"{doc_id_prefix}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        chunks_text = text_splitter.split_text(text)
 
        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for j, chunk_content in enumerate(chunks_text):
            metadatas_to_add.append({"source": doc_id_prefix})
            ids_to_add.append(f"{doc_id_prefix}_chunk_{j}")
            documents_to_add.append(chunk_content)

        chromacollection.add(
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        

# Agregar ideas a la base de datos de ideas previas
def add_idea_to_chromaI(idea: str):
    """Agrega una idea a ChromaDB."""
    
    count = len(list(Path(IDEAPATH).glob("*.txt")))
    idea_id = f"idea_{count}"
    with open(Path(IDEAPATH) / f"{idea_id}.txt", "w", encoding="utf-8") as f:
        f.write(idea)
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=250, 
        chunk_overlap=50, 
        length_function=len,
        is_separator_regex=False,
    )

    # Hacer en chunks para mejor busqueda
    chunks_text = text_splitter.split_text(idea)
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    for j, chunk_content in enumerate(chunks_text):
        metadatas_to_add.append({"source": idea_id})
        ids_to_add.append(f"{idea_id}_chunk_{j}")
        documents_to_add.append(chunk_content)
    chromaIcollection.add(
        documents=documents_to_add,
        metadatas=metadatas_to_add,
        ids=ids_to_add
    )



def search_to_text(query: str, k: int = 2) -> str:
    """Realiza una búsqueda en DuckDuckGo y devuelve los textos de los resultados."""

    # Realizar la búsqueda y regresa los links
    search_tool = DuckDuckGoSearchResults(output_format="json")
    results = loads(search_tool.run(query, k=k))
    
    # Usando BeautifulSoup para extraer texto de las páginas web (dejando codigo de programacion en formato correcto)
    texts = []
    for result in results:
        url = result['link']
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            page_text = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip() != "" and len(p.get_text().strip()) > 50])
            soup_code = BeautifulSoup(response.text, 'html.parser')
            code_texts = [code.get_text() for code in soup_code.find_all('code')]
            full_text = page_text + "\n" + "\n Formatted Code".join(code_texts)
            texts.append(full_text)
        except Exception as e:
            pass

    return [text for text in texts if text.strip() != ""]

def retrieval_node(state: RAGState) -> RAGState:
    """Busca contexto relevante usando ChromaDB."""
    
    # Obtener la pregunta más reciente
    question = state["messages"][-1].content
    
    # Buscar los 2 chunks más relevantes usando el retriever de Chroma y obetener el texto de source
    docs = chromacollection.query(
        query_texts=[question],
        n_results=2
    )
    with open(TEXTS_DIR+"//"+docs["metadatas"][0][0]["source"] + ".txt", "r", encoding="utf-8") as f:
        source_text = f.read()

    # Formatear el contexto
    context = source_text
    
    return {"context": context}

def generation_node(state: RAGState, llm) -> RAGState:
    """Combina el contexto y la pregunta para generar una respuesta."""
    
    context = state["context"]
    question = state["messages"][-1].content
    
    prompt = f"""Usa el siguiente contexto para responder la pregunta de la mejor manera posible.
    --- CONTEXTO ---
    {context}
    --- PREGUNTA ---
    {question}
    """
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

# Trabajador rapido para tareas específicas
trabajador = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API)

# Funcion principal para generar el plan de trabajo
def boss_generation(pregunta: str) -> dict:
    """Función principal para generar el plan de trabajo."""
    
    # Revisar que regresa rettrieval con la función de búsqueda
    state_rag = RAGState(messages=[HumanMessage(content=pregunta)], context="")
    state_rag = retrieval_node(state_rag)

    # Ver si hay en ChromaI ideas relacionadas muy cercanas
    try:
        docs_ideas = chromaIcollection.query(
            query_texts=[pregunta],
            n_results=1,
        )
        if docs_ideas['distances'][0][0] < 0.8:
            with open(IDEAPATH+"//"+docs_ideas["metadatas"][0][0]["source"] + ".txt", "r", encoding="utf-8") as f:
                idea_text = f.read()
            prompt_idea = f"""Usa la siguiente idea relacionada para ayudar a responder la pregunta de la mejor manera posible.
            --- IDEA RELACIONADA ---
            {idea_text}
            --- PREGUNTA ---
            {pregunta}
            """
            messages_idea = [HumanMessage(content=prompt_idea)]
            response_idea = trabajador.invoke(messages_idea)
            return {"final_answer": response_idea.content}
    except Exception as e:
        print("No se encontraron memorias relacionadas")

    # Generar el plan de trabajo usando el boss LLM si no hay idea pre hecha
    prompt_boss = f"""Es necesario que generes un plan de trabajo para responder la siguiente pregunta:
    {pregunta}
    Se te proporciona el siguiente contexto relevante:
    --
    {state_rag['context']}
    --
    Si el contexto no es suficiente, deberas indicarlo en el json resultante.
    Divide el trabajo en varias tareas que puedan ser resueltas por trabajadores especializados.
    Para la buena union de las respuestas, indica palabras clave que ayuden a unir las respuestas de los trabajadores.
    Proporciona la respuesta en formato JSON con la siguiente estructura:""" + """
    {
        "tasks": ["task1 in detail", "task2 in detail", "..."],
        "keywords": ["keyword1", "keyword2", "..."]
    }"""
    messages_boss = [HumanMessage(content=prompt_boss)]
    response_boss = boss.invoke(messages_boss)


    full_dict = loads(response_boss.content)

    # Ejecutar los trabajadores para cada tarea
    return worker_execution(full_dict, pregunta)
    
def worker_execution(dicty:dict, pregunta:str) -> dict:
    """Ejecuta los trabajadores para cada tarea"""
    results = {"ntasks" : len(dicty["tasks"]), "responses": [], "evaluation": []}
    progy = Progress() # Barra de progreso usando Rich
    progy.start()
    tasky = progy.add_task("[green]Processing tasks...", total=results["ntasks"])
    for i, task in enumerate(dicty["tasks"]):
        # Busqueda de contexto relevante para la tarea en la Web
        add_chroma_documents(search_to_text(task, k=1))
        # Obtener contexto relevante para la tarea
        state_rag = RAGState(messages=[HumanMessage(content=task)], context="")
        state_rag = retrieval_node(state_rag)
        context = state_rag["context"]
        prompt_worker = f"""Usa el siguiente contexto para responder la tarea de la mejor manera posible.
        --- CONTEXTO ---
        {context}
        --- TAREA ---
        {task}.
        ---
        Ademas utiliza algunas de las siguientes palabras clave ya que tu respuesta sera unida con las de otros trabajadores: {', '.join(dicty['keywords'])}
                                                                                                                               
        """
        messages_worker = [HumanMessage(content=prompt_worker)]
        response_worker = trabajador.invoke(messages_worker)
        results["responses"].append(response_worker.content)
        progy.update(tasky, advance=1)
        
    progy.stop()
    return combine_answers(results, pregunta)

def combine_answers(worker_results: dict, pregunta: str) -> dict:
    """Combina las respuestas de los trabajadores en una respuesta final."""
    global contexty
    # Unir las respuestas usando el joiner LLM
    prompt_combiner = f"""Usa las siguientes respuestas de varios trabajadores para generar una respuesta final a la pregunta: {pregunta}
{"\n\n--".join(worker_results["responses"])}"""
    messages_combiner = [SystemMessage(content="CONTEXTO: " + contexty), HumanMessage(content=prompt_combiner)]
    response_combiner = joiner.invoke(messages_combiner)
    add_idea_to_chromaI(f"Idea trabajada anteriormente: \n PREGUNTA: \n {pregunta} \n PENSAMIENTOS: \n {'\nPensamiento:'.join(worker_results['responses'])} \n RESPUESTA FINAL: \n {response_combiner.content}")
    return {"final_answer": response_combiner.content}

# Loop principal para interacción con el usuario
ans = ""
contexty = ""
while ans != "exit":
    ans = Prompt.ask("\nIngresa tu pregunta ('exit' para salir)")
    if ans != "exit":
        final_response = boss_generation(ans)
        contexty = contexty + "\n\nPregunta de Usuario" + ans + "\nRespuesta: " + final_response["final_answer"]
        print(f"\n[bold green]Respuesta Final:[/bold green] {final_response['final_answer']}")





