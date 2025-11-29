import os
import json
import logging
from typing import List, Tuple, Annotated, TypedDict, Dict, Any, Optional, Literal
from datasets import load_dataset
import pickle
import faiss
import time
import pandas as pd
import networkx as nx
import tiktoken
from Levenshtein import distance as lev_distance
import wikipedia
from Bio import Entrez
import requests
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download, InferenceClient
from datasets import load_dataset
import zipfile
import operator
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages



encoding = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DS_API_KEY = st.secrets.get("DS_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN")
try:
    feature_client = InferenceClient(
        provider="hf-inference",
        api_key=os.environ["HF_TOKEN"],
    )
    similarity_client = InferenceClient(
        provider="hf-inference",
        api_key=os.environ["HF_TOKEN"],
    )
    rerank_client = InferenceClient(
        provider="auto",
        api_key=os.environ["HF_TOKEN"],
    )
    
    logger.info("HuggingFace InferenceClients initialized successfully")
except Exception as e:
    logger.error(f"HuggingFace client initialization ERROR: {e}")
    feature_client = None
    similarity_client = None
    rerank_client = None
ENTREZ_EMAIL = st.secrets.get("ENTREZ_EMAIL")

Entrez.email = ENTREZ_EMAIL
MAX_TOKENS = 128000


@st.cache_resource(show_spinner="Loading data resources...")
def load_all_resources():
    try:
        if not HF_TOKEN:
            st.error("❌ HF_TOKEN not found, please configure in Streamlit Secrets")
            st.stop()
        
        os.makedirs("data", exist_ok=True)
        files_to_download = [
            "faiss_node+desc.index",
            "faiss_node+desc.pkl",
            "faiss_node.index",
            "faiss_node.pkl",
            "faiss_triple3.index",
            "faiss_triple3.pkl",
            "kg.gpickle",
            "cengyongming.csv"
        ]
        
        st.info("📦Downloading data files...")
        
        for filename in files_to_download:
            downloaded_path = hf_hub_download(
                repo_id="achenyx1412/DGADIS",
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
                cache_dir="./cache"
            )
            
            import shutil
            shutil.copy(downloaded_path, f"data/{filename}")
        
        st.success("✅ All files downloaded.")
        
    except Exception as e:
        st.error(f"❌ Error loading resource: {str(e)}")
        with st.expander("🔍 Full error message"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()
# ======================== Global variables ========================
faiss_indices = {}
metadata = {}
graph = None
merged_data = None
tokenizer = None
model = None
bi_tokenizer = None
bi_model = None
cross_tokenizer = None
cross_model = None
llm = None
name_search_engine = None
compiled_graph = None

# ======================== State definition ========================
class MyState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    entity: list
    target_label: list
    neo4j_retrieval: dict
    llm_answer: str
    pubmed_search: str
    wikipedia_search: str
    api_search: str
    route: str
    sufficient_or_insufficient: str
    interaction: str
    summarized_query: str
    parsed_query: str
    user_reply: str
    ai_message: str
    need_user_reply: bool
    user_reply_text: str


label_list = [
    "Topography and Morphology", "Chemicals, Drugs, and Biological Products",
    "Physical Agents, Forces, and Medical Devices", "Diseases and Diagnoses",
    "Procedures", "Living Organisms", "Social Context", "Symptoms, Signs, and Findings",
    "Disciplines", "Relevant Persons and Populations", "Numbers",
    "Physiological, Biochemical, and Molecular Mechanisms", "Scientific Terms and Methods",
    "Others"
]


# ======================== Name search engine ========================
class NameSearchEngine:
    def __init__(self, merged_data_df):
        self.merged_data = merged_data_df
        self.merged_data['原名列表'] = self.merged_data['原名列表'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        self.current_to_old_map = {}
        self.all_names_map = {}
        
        for _, row in self.merged_data.iterrows():
            现用名 = row['现用名']
            原名列表 = row['原名列表']
            self.current_to_old_map[现用名] = 原名列表
            self.all_names_map[现用名] = 现用名
            for 原名 in 原名列表:
                self.all_names_map[原名] = 现用名
        
        self.searchable_names = list(self.all_names_map.keys())
    
    def calculate_similarity(self, str1, str2):
        if not str1 or not str2:
            return 0.0
        edit_distance = lev_distance(str1, str2)
        max_length = max(len(str1), len(str2))
        if max_length == 0:
            return 1.0
        return max(0.0, 1 - (edit_distance / max_length))
    
    def search(self, query, topk=5, similarity_threshold=0.3):
        query = str(query).strip()
        if not query:
            return []
        results = []
        for name in self.searchable_names:
            similarity = self.calculate_similarity(query, name)
            if similarity >= similarity_threshold:
                现用名 = self.all_names_map[name]
                results.append({
                    'searched_name': 现用名,
                    'similarity': similarity
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return [r['searched_name'] for r in results[:topk]]
# ======================== Helper functions ========================
def _extract_json_from_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return {}
    return {}
def embed_entity(entity_text: str):
    if not feature_client:
        raise ValueError("Feature extraction client not initialized")
    
    try:
        result = feature_client.feature_extraction(
            entity_text,
            model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        )
        if hasattr(result, 'shape'):
            embedding = result
        else:
            embedding = np.array(result)
        if len(embedding.shape) > 1:
            embedding = np.mean(embedding, axis=0)
            
        return embedding
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        raise
def search_pubmed(pubmed_query: str, max_results: int = 3) -> str:
    try:
        handle = Entrez.esearch(db="pubmed", term=pubmed_query, retmax=max_results)
        record = Entrez.read(handle)
        id_list = record["IdList"] if "IdList" in record else []
        print(f"🔍 Query: {pubmed_query} → Found {len(id_list)} results")

        if not id_list:
            return "no articles on pubmed"

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)

        results = []
        for article in records["PubmedArticle"]:
            abstract_parts = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
            abstract_text = " ".join(abstract_parts)
            doi = None
            for id_item in article["PubmedData"]["ArticleIdList"]:
                if id_item.attributes.get("IdType") == "doi":
                    doi = str(id_item)
            results.append({"abstract": abstract_text, "doi": doi})
        return results
    except Exception as e:
        return f"error in pubmed: {e}"


def search_wikipedia(wikipedia_query, max_chars_per_entity=500) -> str:
    try:
        try:
            summary = wikipedia.summary(wikipedia_query, auto_suggest=False)
        except Exception:
            candidates = wikipedia.search(wikipedia_query, results=5)
            summary = None
            for cand in candidates:
                try:
                    summary = wikipedia.summary(cand, auto_suggest=False)
                    break
                except Exception:
                    continue
            if summary is None and candidates:
                try:
                    summary = wikipedia.summary(candidates[0], auto_suggest=True)
                except Exception:
                    summary = None
            if summary is None:
                raise RuntimeError(f"No viable Wikipedia page found for '{wikipedia_query}'")
        clipped = (summary[:max_chars_per_entity] + "...") if len(summary) > max_chars_per_entity else summary
        return f"### {wikipedia_query}\n{clipped}"
    except Exception as e:
        logger.warning(f"error in Wikipedia: {e}")
        return f"error in Wikipedia: {str(e)}"

# ======================== Prompt ========================
LLM = ChatOpenAI(model="deepseek-reasoner",api_key=DS_API_KEY,base_url="https://api.deepseek.com/v1",temperature=0.0)
extract_prompt_en = PromptTemplate(
    input_variables=["query", "label_list"],
    template="""
You are a highly specialized AI assistant for dental query analysis.  
Your **ONLY** task is to (1) summarize and refine the given query for clarity, (2) extract structured entities and intent labels, and (3) judge whether the question provides sufficient information — nothing else.

---

### LANGUAGE POLICY — STRICTLY ENFORCED
- The **input question may be in ANY language** (e.g., Chinese, Spanish, etc.).
- You **MUST translate the entire question into precise, professional English in dental medicine** before processing.
- **ALL extracted entities (both compound and atomic) MUST be in English**, even if the original term was not.
- **DO NOT preserve or output any non-English text.**

---

### TASK 0: Query Summarization and Refinement

Because the input query may include multiple dialogue turns or excessive context,  
you must first perform **concise summarization** of the user's true question before analysis.

Steps:
1. Carefully read the entire input ({query}).
2. Extract only the medically meaningful and question-relevant part.
3. Rephrase it into **a single clear, short, and precise English question**.
   - Example: From “Earlier I asked about gingivitis, and now I want to know what medicines are used for it?” →  
     Summarized query: "What medications are used to treat gingivitis?"

After summarization, all following tasks (entity extraction, labeling, sufficiency judgment)  
MUST be based **only on this summarized query**.

---

### TASK 1: Entity Extraction (MUST be in English)

Extract exactly two types of entities:

1. **compound** (1–2 items max):
   - The full meaningful phrase **as it appears in the translated English question**.
   - Example: If the question is “What is the treatment of gingivitis?” extract → ["gingivitis treatment"]
   - Preserve modifiers: e.g., “soft impression material” → ["soft impression material"]
   - Must be in English.

2. **atomic** (1–3 items max):
   - **ONLY the core biomedical/dental entity name** — must be a concrete, specific term.
   - Examples: "gingivitis", "dental implant", "composite resin"
   - **FORBIDDEN**: generic words like "treatment", "symptom", "complication", "method", "index", "effect".
   - If the compound is "gingivitis treatment" → atomic must be ["gingivitis"], NOT ["treatment"].
   - Must be in English.

If no valid medical entity exists → return empty lists: "compound": [], "atomic": []

---

### TASK 2: Intent Label Selection

- Select 1–3 **most relevant** labels from this list:
{label_list}

- Labels must **exactly match** the provided options.
- Choose only labels that correspond to **node types needed to answer the question**.
- Do NOT invent, modify, or translate label names.

---

### TASK 3: Information Sufficiency Judgment

After analyzing the refined question and extracted entities:

- If the question **contains enough detail** for a meaningful medical/dental answer, set  
  "sufficient_or_insufficient": "sufficient"

- If the question is **ambiguous, missing context, or requires clarification**, set  
  "sufficient_or_insufficient": "insufficient"  
  and in "interaction", **clearly state what additional information the user needs to provide**.  
  Example: "interaction": "Please specify which treatment method or patient condition you are asking about."

If information is sufficient, output "interaction": "nan".

---

### OUTPUT FORMAT — NON-NEGOTIABLE

Output **ONLY** a single, valid JSON object, strictly following this schema:

{{"summarized query": "string (the summarized English question)",
  "entity": {{
    "compound": [string],
    "atomic": [string]
  }},
  "target_label": [string],
  "sufficient_or_insufficient": "sufficient" | "insufficient",
  "interaction": "nan" | "string (interaction message)"
}}

All strings in English.  
No explanations, no markdown, no notes.

---

### EXAMPLES (Follow Exactly)

**Example 1 — Sufficient Information**  
Question: "I have gingivitis. I feel painful. What is the treatment?"  
Output:  
{{"summarized_query": "What is the treatment of gingivitis?",
  "entity": {{
    "compound": ["gingivitis treatment"],
    "atomic": ["gingivitis"]
  }},
  "target_label": ["Procedures", "Chemicals, Drugs, and Biological Products"],
  "sufficient_or_insufficient": "sufficient",
  "interaction": "nan"
}}

**Example 2 — Insufficient Information**  
Question: "What is the best treatment?"  
Output:  
{{"summarized_query": "What is the best treatment?",
  "entity": {{
    "compound": ["treatment"],
    "atomic": []
  }},
  "target_label": ["Procedures"],
  "sufficient_or_insufficient": "insufficient",
  "interaction": "Please specify which disease or condition you are referring to."
}}

---

### FINAL INSTRUCTION

**Question to process:**  
{query}

→ Output ONLY the JSON. No other text.
"""
)
chain1 = extract_prompt_en | LLM
extract_prompt_en_t = PromptTemplate(
    input_variables=["query"],
    template="""
You are a highly specialized AI assistant for dental query analysis. Your ONLY task is to extract a structured SPO triple (subject–predicate–object) from a dental-related question — nothing else.

---

### LANGUAGE POLICY — STRICTLY ENFORCED
- The input question may be in ANY language (e.g., Chinese, Spanish, etc.).
- You MUST translate the entire question into precise, professional English in dental medicine before processing.
- ALL extracted entities and relations MUST be in English, even if the original term was not.
- DO NOT preserve or output any non-English text.

---

### TASK: SPO Triple Extraction

Your task is to convert the question into a concise factual statement (triple) using the following structure:

(SUBJECT, PREDICATE, OBJECT)

#### Rules:
1. The SUBJECT should include any condition, disease, patient group, or object implied in the question.
   - e.g., "children with dental trauma", "impression material", "implant restoration".

2. The PREDICATE should summarize the core intent or relationship implied by the question.
   - Common examples:
     "has treatment", "has complication", "is measured by", "is caused by", "is indicated for", "has preventive method", "has material".
   - The predicate should be neutral, not in question form (avoid “what”, “how”, “which” etc.).

3. The OBJECT should remain as "unknown".
   - This means you do not predict the answer type (e.g., “treatment method” or “index”), only mark it as "unknown". 
   - The purpose is to represent the question as a knowledge triple skeleton. 

4. If the subject already includes the condition modifier (like “for children”), integrate it directly, e.g.:
   - “Children dental trauma has treatment”
   - “Impression material has measurement index”

---

### OUTPUT FORMAT — STRICTLY ENFORCED

Output ONLY one valid JSON object:

{{
  "triple": {{
    "subject": "string",
    "predicate": "string",
    "object": "unknown"
  }}
}}

No markdown, no explanations, no extra text.

---

### EXAMPLES

Example 1  
Question: "What is the treatment of gingivitis?"  
Output:
{{
  "triple": {{
    "subject": "gingivitis",
    "predicate": "has treatment",
    "object": "unknown"
  }}
}}

Example 2  
Question: "What are the complications of implant restoration?"  
Output:
{{
  "triple": {{
    "subject": "implant restoration",
    "predicate": "has complication",
    "object": "unknown"
  }}
}}

Example 3  
Question: "印模材料凝固后，其软度通常用什么指标表示？"  
(Translated: "After impression material solidifies, what index expresses its softness?")  
Output:
{{
  "triple": {{
    "subject": "impression material",
    "predicate": "has measurement index",
    "object": "unknown"
  }}
}}

Example 4  
Question: "对于儿童的牙外伤应该如何治疗？"  
Output:
{{
  "triple": {{
    "subject": "children dental trauma",
    "predicate": "has treatment",
    "object": "unknown"
  }}
}}

---

### FINAL INSTRUCTION

Question to process:
{query}

→ Output ONLY the JSON triple above. Nothing else.
"""
)
chain1_t = extract_prompt_en_t | LLM
knowledge_router_prompt_en = PromptTemplate(
    input_variables=["neo4j_retrieval", "query"],
    template="""
You are an expert dental medicine AI router specialized in evaluating knowledge sufficiency and generating targeted retrieval queries.

---

### OBJECTIVE
Your function is **NOT** to answer the user's question directly.  
Instead, you evaluate whether the provided **Knowledge Graph Context** contains enough information to fully and accurately answer the question.  
If not, you will identify the **specific knowledge gaps** and write **search queries** to retrieve only the missing parts — **do NOT discard or ignore the existing context**.

---

### INPUTS

**Knowledge Graph Context:**
{neo4j_retrieval}

**User's Question:**
{query}

---

### INSTRUCTIONS

1. **Carefully analyze** the Knowledge Graph Context and the User's Question together.  
   - Consider what information is already covered by the Knowledge Graph Context.  
   - Identify what information is **missing** (the “knowledge gaps”) that prevents a complete answer.

2. **If the context is sufficient**, respond with:
   - `"answer": "sufficient_knowledge"`
   - Leave both `"pubmed_search"` and `"wikipedia_search"` as empty strings.

3. **If the context is insufficient**, respond with:
   - `"answer": "lack_knowledge"`
   - Generate **two concise and high-quality retrieval queries** focused ONLY on the missing knowledge:
     - `"pubmed_search"`: a Boolean-style scientific query suitable for PubMed  
       (use terms, synonyms, and AND/OR operators; 5–12 words total)
     - `"wikipedia_search"`: a natural language query suitable for Wikipedia  
       (short, clear, and human-readable; 3–8 words total)

   **Do not repeat or rephrase existing context.**  
   Your goal is to complement what is missing — not replace the Knowledge Graph Context.

4. **Do not include explanations, markdown, or reasoning text.**  
   Output only a **valid JSON** object.

---

### OUTPUT FORMAT

Your response must strictly follow this structure:

{{
  "answer": "sufficient_knowledge" | "lack_knowledge",
  "pubmed_search": "string",
  "wikipedia_search": "string"
}}

---

### EXAMPLES

**Example 1 — Context Sufficient**
Question: "What is the treatment of gingivitis?"  
Knowledge Graph Context already includes detailed information about gingivitis treatments.  
Output:
{{
  "answer": "sufficient_knowledge",
  "pubmed_search": "",
  "wikipedia_search": ""
}}

**Example 2 — Context Insufficient**
Question: "What are the molecular mechanisms of peri-implantitis?"  
Knowledge Graph Context only includes definitions and symptoms.  
Output:
{{
  "answer": "lack_knowledge",
  "pubmed_search": "(peri-implantitis) AND (molecular mechanism OR inflammatory pathway)",
  "wikipedia_search": "molecular mechanisms of peri-implantitis"
}}

---

### FINAL RULE
Provide only the JSON object as your final response — nothing else.
"""
)
chain2 = knowledge_router_prompt_en | LLM

final_answer_prompt_en = PromptTemplate(
    input_variables=["query", "neo4j_retrieval", "api_search_result"],
    template = """
You are a highly authoritative dental medicine AI assistant. Respond with the tone and reasoning style of an experienced clinical dentist. Always provide clear, confident, and expert-level explanations in English.

Essay Question:
{query}

Knowledge Graph Information:
{neo4j_retrieval}

External Search (PubMed, Wikipedia):
{api_search_result}

Requirements:

Prioritize the provided context when forming your answer.

When context is insufficient, rely on your own expert dental knowledge. Provide a confident, clinically grounded explanation.

Maintain a professional, precise, and authoritative dental-specialist tone.

Always include a Source field at the end:

Use the knowledge-graph edge’s chunk_id when applicable.

Use the PubMed article’s DOI when applicable.

Use "wikipedia" when drawing from Wikipedia.

Use "LLM_database" when the answer is based on your internal professional knowledge.
Your answer must be in English
"""
)
chain3 = final_answer_prompt_en | LLM
# ======================== Processing nodes ========================
def parse_query(state: MyState):
    logger.info("---NODE: parse_query---")
    
    # Get the latest user messages
    messages = state.get("messages", [])
    if not messages:
        return {
            "sufficient_or_insufficient": "insufficient", 
            "interaction": "No query provided. Please describe your dental issue."
        }
    
    # Get the content of the last user message
    user_query = ""
    for message in reversed(messages):
        if hasattr(message, 'content'):
            user_query = message.content
            break
    
    print(f"parse_query: {user_query}")
    
    if not user_query or len(user_query.strip()) < 3:
        return {
            "sufficient_or_insufficient": "insufficient",
            "interaction": "Your query is too short. Please provide more details about your dental issue."
        }
    
    query_str = user_query
    parse_outcome = chain1.invoke({"query": query_str, "label_list": "\n".join(label_list)})
    parse_outcome_t = chain1_t.invoke({"query": query_str})
    
    try:
        parsed_text = getattr(parse_outcome, "content", str(parse_outcome)).strip()
        parsed_json = _extract_json_from_text(parsed_text)
        print(f"parse_json:{parsed_json}")
        
        entity_compound_atomic = parsed_json.get("entity", {})
        entity_compound = entity_compound_atomic.get("compound", [])
        entity_atomic = entity_compound_atomic.get("atomic", [])
        summarized_query = parsed_json.get("summarized_query")
        target_label = parsed_json.get("target_label", [])
        sufficient_or_insufficient = parsed_json.get("sufficient_or_insufficient", "sufficient")
        interaction = parsed_json.get("interaction", "You need to provide more information.")
        
        entity_name = []
        entity_name.extend(entity_compound)
        entity_name.extend(entity_atomic)
        entity_name = entity_name[:6]
        
        parsed_text_t = getattr(parse_outcome_t, "content", str(parse_outcome_t)).strip()
        parsed_json_t = _extract_json_from_text(parsed_text_t)
        parsed_triple = parsed_json_t.get("triple", {})
        triple_subject = parsed_triple.get("subject","")
        triple_predicate = parsed_triple.get("predicate","")
        triple_object = parsed_triple.get("object","")
        parsed_query = f"{triple_subject} {triple_predicate} {triple_object} "

        logger.info(f"entity_name={entity_name},target_label={target_label}")
        return {
            "entity": entity_name,
            "target_label": target_label,
            "summarized_query": summarized_query,
            "sufficient_or_insufficient": sufficient_or_insufficient,
            "interaction": interaction,
            "parsed_query": parsed_query
        }
    except Exception as e:
        logger.warning(f"JSON failed: {e}")
        return {
            "sufficient_or_insufficient": "insufficient",
            "interaction": f"I encountered an error processing your query. Please try again or provide more details. Error: {str(e)}"
        }

def handle_user_input(state: dict, user_reply_text = None):
    print("---NODE: user_input---")
    
    interaction_content = state.get(
        "interaction",
        "Your question is not informative enough. Please describe the problem in more detail."
    )

    print(f"Interaction content: {interaction_content}")
    print(f"User reply text: {user_reply_text}")

    # Case 1: No user input yet (set stop state)
    if not user_reply_text:
        print("STOPPING FLOW - Waiting for user input...")
        return {
            "ai_message": interaction_content,
            "need_user_reply": True,               # Tell the frontend need_user_reply
            "flow_stopped": True,
            # messages are not returned, and the original conversation is kept
        }

    # Case 2: User input has been received (clear stop status, continue process)
    print(f"RESUMING FLOW - Received user reply: {user_reply_text}")
    return {
        "ai_message": interaction_content,
        "need_user_reply": False,
        "flow_stopped": False,
        "messages": [HumanMessage(content=user_reply_text)],  # Add a new user message
        "user_reply": user_reply_text,
    }

def whether_to_interact(state):
    print("---EDGE: whether_to_interact---")
    interaction = state.get("sufficient_or_insufficient")
    print(f"sufficient_or_insufficient: {interaction}")
    
    if interaction == "insufficient":
        print("Decision: Insufficient information, need user input.")
        return "user_input"
    elif interaction == "sufficient":
        print("Decision: Information is sufficient to continue.")
        return "neo4j_retrieval"
    else:
        print(f"Decision: Unknown state '{interaction}', routing to neo4j_retrieval as default.")
        return "neo4j_retrieval"

def after_user_input(state):
    print("---EDGE: after_user_input---")
    print(f"user_reply_text: {state.get('user_reply_text')}")
    print(f"need_user_reply: {state.get('need_user_reply')}")
    
    # If a user response is needed (waiting for input), end the current flow
    if state.get("need_user_reply"):
        print("Decision: Waiting for user input, ending flow.")
        return "end"
    
    # If user replies, continue the process
    if state.get("user_reply_text"):
        print("Decision: User provided input, continuing to parse_query.")
        return "parse_query"
    print("Decision: No user input needed, ending flow.")
    return "end"

def neo4j_retrieval(state: MyState, resources):
    idx1 = faiss.read_index("data/faiss_node+desc.index")
    with open("data/faiss_node+desc.pkl", "rb") as f:
        meta1 = pickle.load(f)
    idx2 = faiss.read_index("data/faiss_node.index")
    with open("data/faiss_node.pkl", "rb") as f:
        meta2 = pickle.load(f)
    idx3 = faiss.read_index("data/faiss_triple3.index")
    with open("data/faiss_triple3.pkl", "rb") as f:
        meta3 = pickle.load(f)
    with open("data/kg.gpickle", "rb") as f:
        G = pickle.load(f)
    logger.info("---NODE: neo4j_retrieval---")
    #user_query = [message.content for message in state["messages"] if hasattr(message, 'content')]
    #query_str = user_query[0]
    #query_text = " ".join(query_str) if isinstance(query_str, list) else str(query_str)
    query_text = state.get("summarized_query")
    entity_list = state.get("entity", []) or []
    target_labels = state.get("target_label", []) or []
    parsed_query = state.get("parsed_query", "") or ""
    topk = 5
    depth = int(os.getenv("GRAPH_SEARCH_DEPTH", "2"))

    if not entity_list or not target_labels:
        return {"neo4j_retrieval": []}

    path_kv: Dict[str, str] = {}
    for entity in entity_list:
        try:
            entity_embedding2 = embed_entity(parsed_query).reshape(1, -1)
            D, I = idx3.search(entity_embedding2, 5)
            candidate_triples = []
            for idx in I[0]:
                idx_int = int(idx)
                if 0 <= idx_int < len(meta3):
                    candidate_triples.append(meta3[idx_int])
                else:
                    logger.warning(f"Index {idx_int} out of range for meta3 (len={len(meta3)})")
                    
            logger.info(f"Found {len(candidate_triples)} candidate triples:{candidate_triples[:1]}")
            cand_info = [{
            "head": cand.get("head", ""),
            "head_desc": cand.get("head_desc", ""),
            "rel": cand.get("rel", ""),
            "rel_desc": cand.get("rel_desc", ""),
            "rel_id": cand.get("rel_id", ""),
            "tail": cand.get("tail", ""),
            "tail_desc": cand.get("tail_desc", "")}
            for cand in candidate_triples]
            
            entity_embedding = embed_entity(entity).reshape(1, -1)
            candidates1 = []
            try:
                D1, I1 = idx1.search(entity_embedding, topk)
                for idx in I1[0]:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(meta1):
                        candidates1.append(meta1[idx_int])
                    else:
                        logger.warning(f"Index {idx_int} out of range for meta1")
                logger.info(f"idx1 returned {len(candidates1)} candidates:{candidates1[:1]}")
            except Exception as e:
                logger.warning(f"idx1 search failed: {str(e)}")

            candidates2 = []
            try:
                D2, I2 = idx2.search(entity_embedding, topk)
                for idx in I2[0]:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(meta2):
                        candidates2.append(meta2[idx_int])
                    else:
                        logger.warning(f"Index {idx_int} out of range for meta2")
                logger.info(f"idx2 returned {len(candidates2)} candidates:{candidates2[:1]}")
            except Exception as e:
                logger.warning(f"idx2 search failed: {str(e)}")
                
            cengyongming_df = pd.read_csv("data/cengyongming.csv")
            search_engine = NameSearchEngine(cengyongming_df)

            cand_names3 = search_engine.search(entity, topk=topk)
            name_list = []
            for cand in candidates1:
                cand_id = cand["id"]
                cand_name = cand["name"]
                if cand_name not in G:
                    logger.warning(f"[WARN]  {cand_name}) not in kg")
                    continue
                if cand_name not in name_list:
                    name_list.append(cand_name)
                    logger.info(f"[INFO] node+desc {cand_name}) added to name_list")
            for cand in candidates2:
                cand_id = cand["id"]
                cand_name = cand["name"]
                if cand_name not in G:
                    logger.warning(f"[WARN]  {cand_name}) not in kg")
                    continue
                if cand_name not in name_list:
                    name_list.append(cand_name)
                    logger.info(f"[INFO] node {cand_name}) added to name_list")
            for cand_name in cand_names3:
                if cand_name not in G:
                    logger.warning(f"[WARN]  {cand_name}) not in kg")
                    continue
                if cand_name not in name_list:
                    name_list.append(cand_name)
                    logger.info(f"[INFO] name_search {cand_name}) added to name_list")
            for cand_name in name_list:                            
                try:
                    for target_label in target_labels:
                        neighbors = [
                            n for n, data in G.nodes(data=True)
                            if target_label in data.get("labels", [])
                        ]
                        for nbr in neighbors:
                            if nx.has_path(G, cand_name, nbr):
                                path = nx.shortest_path(G, source=cand_name, target=nbr)
                                if len(path) - 1 <= depth:
                                    parts_key = []
                                    parts_val = []
                                    for i, node in enumerate(path):
                                        n_data = G.nodes[node]
                                        n_name = n_data.get("name", "")
                                        n_prop = json.dumps(
                                            {k: v for k, v in n_data.items() if k in ["description"]},
                                            ensure_ascii=False
                                        )

                                        if i == 0:
                                            parts_val.append(f"[{n_name}:{n_prop}]")
                                        else:
                                            prev = path[i - 1]
                                            edge_data = G.get_edge_data(prev, node) or {}
                                            rel_type = edge_data.get("type", "")
                                            rel_src = edge_data.get("chunk_id", "")
                                            rel_text = edge_data.get("original_text", "")

                                            parts_key.append(f"{rel_text}")
                                            parts_val.append(f"--[{rel_type}:{rel_text}]-->[{n_name}:{n_prop}]")

                                    path_key = ";".join(parts_key)
                                    path_value = "".join(parts_val)

                                    if path_key not in path_kv:
                                        path_kv[path_key] = path_value
                except Exception as e:
                    logger.warning(f"[WARN] BFS for candidate {cand_name} error: {e}")
                    continue
            for i in cand_info:
                path_key = f"{i['rel_desc']}"
                path_value = f"[{i['head']}:{i['head_desc']}]--[{i['rel']}:{i['rel_desc']}]-->[{i['tail']}:{i['tail_desc']}]"
                if path_key not in path_kv:
                    path_kv[path_key] = path_value

        except Exception as e:
            logger.warning(f"'{entity}' failed in faiss: {e}")
            logger.debug(f"Entity: {entity}, Embedding shape: {entity_embedding.shape if 'entity_embedding' in locals() else 'N/A'}")
            continue

    try:
        if not similarity_client:
            raise ValueError("HuggingFace client not initialized")
            
        path_keys = list(path_kv.keys())
        
        if not path_keys:
            logger.warning("No paths found for reranking")
            return {"neo4j_retrieval": []}
        
        logger.info("Calculating similarity scores using feature extraction...")
        
        # 第一阶段：使用 BAAI/bge-m3 进行初始检索
        query_embedding = similarity_client.feature_extraction(
            query_text,
            model="BAAI/bge-m3",
        )
        if hasattr(query_embedding, 'shape') and len(query_embedding.shape) > 1:
            query_embedding = query_embedding.mean(axis=1).squeeze()
        
        batch_size = 16
        sim_scores = []
        
        for i in range(0, len(path_keys), batch_size):
            batch_keys = path_keys[i:i + batch_size]
            try:
                batch_embeddings = similarity_client.feature_extraction(
                    batch_keys,
                    model="BAAI/bge-m3",
                )
                
                for j, key_embedding in enumerate(batch_embeddings):
                    if hasattr(key_embedding, 'shape') and len(key_embedding.shape) > 1:
                        key_embedding = key_embedding.mean(axis=1).squeeze()
                    
                    similarity = np.dot(query_embedding, key_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(key_embedding)
                    )
                    sim_scores.append(float(similarity))
                    
            except Exception as batch_error:
                logger.warning(f"Batch similarity calculation error: {batch_error}")
                sim_scores.extend([0.0] * len(batch_keys))
        
        scored_paths = list(zip(path_keys, sim_scores))
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        top30 = scored_paths[:30]
        top30_values = [path_kv[pk] for pk, _ in top30]
        logger.info(f"Successfully reranked 30 paths using BAAI/bge-reranker-large")
        return {"neo4j_retrieval": top30_values}

    except Exception as e:
        logger.warning(f"Rerank error: {e}")
        fallback_values = list(path_kv.values())[:50]
        return {"neo4j_retrieval": fallback_values}

def decide_router(state: MyState) -> dict:
    print("---EDGE: decide_router---")
    neo4j_data = state.get("neo4j_retrieval")
    query_string = state.get("summarized_query")
    neo4j_retrieval = json.dumps(neo4j_data, ensure_ascii=False)
    full_prompt = knowledge_router_prompt_en.format(
        neo4j_retrieval=neo4j_retrieval,
        query=query_string
    )
    total_tokens = len(encoding.encode(full_prompt))
    if total_tokens > MAX_TOKENS:
        neo4j_tokens = len(encoding.encode(neo4j_retrieval))
        allowed_for_retrieval = MAX_TOKENS - total_tokens + neo4j_tokens
        truncated_tokens = encoding.encode(neo4j_retrieval)[:allowed_for_retrieval]
        neo4j_retrieval = encoding.decode(truncated_tokens)
        print(f"Router prompt exceeded tokens")
    try:
        router_outcome = chain2.invoke({
            "neo4j_retrieval": neo4j_retrieval,
            "query": query_string
        })
        router_text = getattr(router_outcome, "content", str(router_outcome)).strip()
        parsed_json = _extract_json_from_text(router_text)
        decision = parsed_json.get("answer", "lack_knowledge")
        if "sufficient_knowledge" in decision:
            print("sufficient knowledge,generate answer directly")
            return {"route": "llm_answer"}
        else:
            print("insufficient knowledge, api search")
            pubmed_query = parsed_json.get("pubmed_search", query_string)
            wikipedia_query = parsed_json.get("wikipedia_search", query_string)
            if not pubmed_query:
                print("llm failed to generate pubmed_query")
                pubmed_query = query_string
            if not wikipedia_query:
                print("llm failed to generate wikipedia_query")
                wikipedia_query = query_string

            print(f"pubmed_query: {pubmed_query}")
            print(f"wikipedia_query: {wikipedia_query}")

            return {
                "route": "api_search",
                "pubmed_search": pubmed_query,
                "wikipedia_search": wikipedia_query
            }

    except Exception as e:
        print(f"Router error: {e}")
        return {
            "route": "api_search",
            "pubmed_search": query_string,
            "wikipedia_search": query_string
        }

def api_search(state: MyState) -> dict:
    logger.info("---NODE: api_search---")
    pubmed_query = state.get("pubmed_search")
    wikipedia_query = state.get("wikipedia_search")
    pubmed_results = search_pubmed(pubmed_query)
    wikipedia_results = search_wikipedia(wikipedia_query)
    api_search_result = f"## PubMed Search Results:\n{pubmed_results}\n\n## Wikipedia Search Results:\n{wikipedia_results}"
    logger.info(f"pubmed_results: {pubmed_results[:100]}\nwikipedia_results: {wikipedia_results[:100]}")
    return {"api_search": api_search_result}

def llm_answer(state: MyState):
    print("Answering process")
    neo4j_data = state.get("neo4j_retrieval")
    neo4j_retrieval = json.dumps(neo4j_data, ensure_ascii=False)
    api_search_result = state.get("api_search")
    user_query = [message.content for message in state["messages"]]
    query_string = user_query

    prompt_base = final_answer_prompt_en.format(
        neo4j_retrieval=neo4j_retrieval,
        api_search_result=api_search_result, 
        query=query_string
    )
    base_tokens = len(encoding.encode(prompt_base))
    neo4j_tokens = len(encoding.encode(neo4j_retrieval))

    if MAX_TOKENS < base_tokens:
        allowed_for_neo4j = neo4j_tokens - base_tokens + MAX_TOKENS
        truncated_tokens = encoding.encode(neo4j_retrieval)[:allowed_for_neo4j]
        neo4j_retrieval = encoding.decode(truncated_tokens)
        print(f"Router prompt exceeded tokens")

    final_answer = chain3.invoke({
        "query": query_string,
        "neo4j_retrieval": neo4j_retrieval,
        "api_search_result": api_search_result
    })

    try:
        final_answer_text = getattr(final_answer, "content", str(final_answer)).strip()
        maybe_json = _extract_json_from_text(final_answer_text)
        if maybe_json and isinstance(maybe_json, dict) and "answer" in maybe_json:
            answer_content = maybe_json["answer"]
        else:
            answer_content = final_answer_text
    except Exception as e:
        print(f"final answer error: {e}")
        answer_content = f"final answer error: {e}"
        print(answer_content)
    logger.info(f"Final answer: {answer_content}")
    return {"llm_answer": answer_content }

# ======================== Build graph ========================
def build_graphrag_agent(resources):
    builder = StateGraph(MyState)

    builder.add_node("parse_query", parse_query)
    builder.add_node("user_input", lambda state: handle_user_input(state, state.get("user_reply_text")))
    builder.add_node("neo4j_retrieval", lambda state: neo4j_retrieval(state, resources))
    builder.add_node("decide_router", decide_router)
    builder.add_node("api_search", api_search)
    builder.add_node("llm_answer", llm_answer)

    builder.add_edge(START, "parse_query")
    builder.add_conditional_edges(
        "parse_query",
        whether_to_interact,
        {
            "user_input": "user_input",
            "neo4j_retrieval": "neo4j_retrieval"
        }
    )
    
    builder.add_conditional_edges(
        "user_input",
        after_user_input,
        {
            "parse_query": "parse_query",
            "end": END  # Maps the "end" string to the END constant
        }
    )
    
    builder.add_edge("neo4j_retrieval", "decide_router")
    builder.add_conditional_edges(
        "decide_router",
        lambda state: state["route"],
        {
            "api_search": "api_search",
            "llm_answer": "llm_answer"
        }
    )
    builder.add_edge("api_search", "llm_answer")
    builder.add_edge("llm_answer", END)
    
    return builder.compile()

# ==================== Streamlit UI ====================
resources = load_all_resources()
graph = build_graphrag_agent(resources)

st.title("DGADIS - Dental Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.graph_state:
    st.sidebar.subheader("Debug Info")
    st.sidebar.write(f"Graph State Keys: {list(st.session_state.graph_state.keys())}")
    st.sidebar.write(f"Has llm_answer: {'llm_answer' in st.session_state.graph_state}")
    if 'llm_answer' in st.session_state.graph_state:
        st.sidebar.write(f"llm_answer length: {len(str(st.session_state.graph_state.get('llm_answer', '')))}")
    st.sidebar.write(f"Has need_user_reply: {st.session_state.graph_state.get('need_user_reply', False)}")

if prompt := st.chat_input("What is your dental question?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.graph_state is None:
        inputs = {"messages": [HumanMessage(content=prompt)]}
        print("=== FIRST CALL ===")
    else:
        inputs = dict(st.session_state.graph_state)
        inputs["user_reply_text"] = prompt
        inputs["need_user_reply"] = False
        print("=== CONTINUATION CALL with user_reply_text ===")
    
    try:
        new_state = graph.invoke(inputs)
        st.session_state.graph_state = new_state
        
        print("=== GRAPH RESPONSE ===")
        print(f"All keys in new_state: {list(new_state.keys())}")
        print(f"need_user_reply: {new_state.get('need_user_reply')}")
        print(f"ai_message: {new_state.get('ai_message')}")
        print(f"llm_answer exists: {'llm_answer' in new_state}")
        if 'llm_answer' in new_state:
            print(f"llm_answer type: {type(new_state.get('llm_answer'))}")
            print(f"llm_answer content: {new_state.get('llm_answer')}")
        
        if new_state.get("need_user_reply"):
            ai_message = new_state.get("ai_message", "Please provide more information.")
            st.session_state.messages.append({"role": "assistant", "content": ai_message})
            with st.chat_message("assistant"):
                st.markdown(ai_message)
            print("=== WAITING FOR USER INPUT ===")
            
        elif new_state.get("llm_answer"):
            answer = new_state.get("llm_answer")
            
            if not isinstance(answer, str):
                answer = str(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            st.session_state.graph_state = None
            
            print("=== FLOW COMPLETED ===")
            
        else:
            status_msg = "Processing completed. No final answer generated."
            st.session_state.messages.append({"role": "assistant", "content": status_msg})
            
            with st.chat_message("assistant"):
                st.markdown(status_msg)
                
            st.sidebar.error("No llm_answer found in state")
            st.sidebar.write("Available keys:", list(new_state.keys()))
            if 'messages' in new_state:
                st.sidebar.write("Last few messages:", [str(msg) for msg in new_state['messages'][-2:]])
                
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR: {error_detail}")
        
        error_msg = f"Sorry, an error occurred: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        with st.chat_message("assistant"):
            st.markdown(error_msg)

if st.session_state.messages:
    st.sidebar.subheader("Conversation Status")
    st.sidebar.write(f"Total messages: {len(st.session_state.messages)}")
    st.sidebar.write(f"Graph state active: {st.session_state.graph_state is not None}")

if st.session_state.messages and st.button("Start New Conversation"):
    st.session_state.messages = []
    st.session_state.graph_state = None
    st.rerun()
