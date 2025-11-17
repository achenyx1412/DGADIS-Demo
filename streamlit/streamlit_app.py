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
from huggingface_hub import hf_hub_download

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from datasets import load_dataset
import zipfile



encoding = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DS_API_KEY = st.secrets.get("DS_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN")
ENTREZ_EMAIL = st.secrets.get("ENTREZ_EMAIL")

Entrez.email = ENTREZ_EMAIL
MAX_TOKENS = 128000

# ======================== 加载数据资源 ========================
class HuggingFaceEmbeddingAPI:
    """使用 Hugging Face Inference API 获取 embeddings"""
    
    def __init__(self, model_name: str, api_token: str):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def encode(self, texts, batch_size=8, normalize=True, max_retries=3):
        """
        模拟 sentence-transformers 的 encode 方法
        返回 numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for retry in range(max_retries):
                try:
                    # ✅ 修复：正确的 API 调用格式
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json={
                            "inputs": batch,
                            "options": {
                                "wait_for_model": True,
                                "use_cache": True
                            }
                        },
                        timeout=60  # 增加超时时间
                    )
                    
                    # 调试信息
                    logger.info(f"API Response Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        embeddings = response.json()
                        
                        # ✅ 处理返回格式
                        # HF Feature Extraction API 通常返回: [[emb1], [emb2], ...]
                        if isinstance(embeddings, list):
                            # 确保是二维数组
                            if len(embeddings) > 0 and isinstance(embeddings[0], list):
                                all_embeddings.extend(embeddings)
                            else:
                                logger.error(f"Unexpected embedding format: {type(embeddings[0])}")
                                all_embeddings.extend([[0.0] * 768] * len(batch))
                        else:
                            logger.error(f"Unexpected response format: {type(embeddings)}")
                            all_embeddings.extend([[0.0] * 768] * len(batch))
                        
                        break  # 成功，跳出重试循环
                    
                    elif response.status_code == 503:
                        # 模型正在加载
                        logger.warning(f"Model loading, waiting... (attempt {retry+1}/{max_retries})")
                        st.info(f"⏳ 模型正在加载，请稍候... (尝试 {retry+1}/{max_retries})")
                        time.sleep(10)  # 等待10秒
                        continue
                    
                    else:
                        error_msg = response.text
                        logger.error(f"API Error {response.status_code}: {error_msg}")
                        st.warning(f"API 调用失败 (状态码 {response.status_code})")
                        
                        if retry == max_retries - 1:
                            # 最后一次重试也失败，返回零向量
                            all_embeddings.extend([[0.0] * 768] * len(batch))
                        else:
                            time.sleep(2)  # 等待后重试
                            continue
                        
                except requests.exceptions.Timeout:
                    logger.error(f"Request timeout (attempt {retry+1}/{max_retries})")
                    if retry == max_retries - 1:
                        st.warning(f"API 请求超时")
                        all_embeddings.extend([[0.0] * 768] * len(batch))
                    else:
                        time.sleep(2)
                        continue
                
                except Exception as e:
                    logger.error(f"API 调用异常: {str(e)}")
                    if retry == max_retries - 1:
                        st.warning(f"API 调用异常: {str(e)}")
                        all_embeddings.extend([[0.0] * 768] * len(batch))
                    else:
                        time.sleep(2)
                        continue
        
        # 转换为 numpy array
        if not all_embeddings:
            logger.error("No embeddings collected")
            return np.zeros((len(texts), 768), dtype=np.float32)
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # 归一化（如果需要）
        if normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除以零
            embeddings_array = embeddings_array / norms
        
        return embeddings_array


class HuggingFaceRerankAPI:
    """使用 Hugging Face Inference API 进行重排序"""
    
    def __init__(self, model_name: str, api_token: str):
        self.model_name = model_name
        # ✅ 修复：使用正确的 API 端点
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def predict(self, pairs, max_retries=3):
        """
        pairs: list of [query, passage] pairs
        返回分数列表
        """
        scores = []
        
        for idx, (query, passage) in enumerate(pairs):
            for retry in range(max_retries):
                try:
                    # ✅ 修复：Reranker 模型的正确调用格式
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json={
                            "inputs": {
                                "source_sentence": query,
                                "sentences": [passage]
                            },
                            "options": {
                                "wait_for_model": True,
                                "use_cache": True
                            }
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # ✅ 处理不同的返回格式
                        if isinstance(result, list) and len(result) > 0:
                            # 返回格式: [score1, score2, ...]
                            score = float(result[0])
                        elif isinstance(result, dict):
                            # 返回格式: {"score": 0.xx} 或 {"similarity": 0.xx}
                            score = result.get('score', result.get('similarity', 0.0))
                        elif isinstance(result, (int, float)):
                            score = float(result)
                        else:
                            logger.warning(f"Unexpected rerank result format: {type(result)}")
                            score = 0.0
                        
                        scores.append(score)
                        break
                    
                    elif response.status_code == 503:
                        logger.warning(f"Model loading for rerank (attempt {retry+1}/{max_retries})")
                        if retry < max_retries - 1:
                            time.sleep(10)
                            continue
                        else:
                            scores.append(0.0)
                    
                    else:
                        logger.error(f"Rerank API Error {response.status_code}: {response.text}")
                        if retry == max_retries - 1:
                            scores.append(0.0)
                        else:
                            time.sleep(2)
                            continue
                
                except Exception as e:
                    logger.error(f"Rerank API 调用异常 (pair {idx+1}): {str(e)}")
                    if retry == max_retries - 1:
                        scores.append(0.0)
                    else:
                        time.sleep(2)
                        continue
        
        return scores


@st.cache_resource(show_spinner="Loading data resources...")
def load_all_resources():
    try:
        # --- 1. 检查 TOKEN ---
        if not HF_TOKEN:
            st.error("❌ HF_TOKEN not found, please configure in Streamlit Secrets")
            st.stop()
        
        os.makedirs("data", exist_ok=True)
        
        # --- 2. 下载文件 ---
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

        # --- 初始化模型 API（不下载模型）---
        st.info("🌐 Initializing model API connection...")
        
        # SapBERT API
        sap_api = HuggingFaceEmbeddingAPI(
            model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            api_token=HF_TOKEN,
            debug=True
        )
        st.success("✅ SapBERT API initialized")
        
        # BGE-M3 API
        bi_api = HuggingFaceEmbeddingAPI(
            model_name="BAAI/bge-m3",
            api_token=HF_TOKEN,
            debug=True
        )
        st.success("✅ BGE-M3 API initialized")
        
        # BGE Reranker API
        cross_api = HuggingFaceRerankAPI(
            model_name="BAAI/bge-reranker-v2-m3",
            api_token=HF_TOKEN,
            debug=True
        )
        st.success("✅ BGE Reranker API initialized")
        
        st.success("🎉All resources are loaded!")
        
        return {
            "sap": (None, sap_api),
            "bi": (None, bi_api),
            "cross": (None, cross_api)
        }
        
    except Exception as e:
        st.error(f"❌ Error loading resource: {str(e)}")
        with st.expander("🔍 Full error message"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()
# ======================== 全局变量 ========================
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

# ======================== 状态定义 ========================
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


label_list = [
    "Topography and Morphology", "Chemicals, Drugs, and Biological Products",
    "Physical Agents, Forces, and Medical Devices", "Diseases and Diagnoses",
    "Procedures", "Living Organisms", "Social Context", "Symptoms, Signs, and Findings",
    "Disciplines", "Relevant Persons and Populations", "Numbers",
    "Physiological, Biochemical, and Molecular Mechanisms", "Scientific Terms and Methods",
    "Others"
]


# ======================== 名称搜索引擎 ========================
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
# ======================== 辅助函数 ========================
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
def embed_entity(text: str, api):
    try:
        emb = api.encode([text], normalize=True)

        # emb shape 应该是 (1, dim)
        if isinstance(emb, np.ndarray) and len(emb.shape) == 2:
            return emb[0].astype('float32')

        # 如果错误格式
        raise RuntimeError("API returned unexpected embedding format")

    except Exception as e:
        logger.error(f"embedding error: {e}")
        return np.zeros(768, dtype='float32')
        
def rerank_paths_with_apis(query_text: str, path_kv: dict, bi_api, cross_api):
    """
    使用 API 进行路径重排序
    
    Args:
        query_text: 查询文本
        path_kv: 路径键值对字典
        bi_api: BGE-M3 API 实例
        cross_api: BGE Reranker API 实例
    
    Returns:
        dict: {"neo4j_retrieval": top30_values}
    """
    try:
        # --- 1. 使用 BGE-M3 API 获取 query embedding ---
        st.info("🔍Calculating the query vector...")
        query_emb = bi_api.encode([query_text], normalize=True)  # shape: (1, dim)
        
        # --- 2. 获取所有候选路径的 embeddings ---
        path_keys = list(path_kv.keys())
        
        if not path_keys:
            logger.warning("No path keys to rerank")
            return {"neo4j_retrieval": []}
        
        st.info(f"📊 Processing {len(path_keys)} path candidates...")
        
        # 分批处理候选路径（API 调用）
        batch_size = 32
        all_cand_embs = []
        
        for i in range(0, len(path_keys), batch_size):
            batch = path_keys[i:i + batch_size]
            st.text(f"⏳ Processing batch {i//batch_size + 1}/{(len(path_keys)-1)//batch_size + 1}...")
            
            # 调用 API 获取 embeddings
            batch_embs = bi_api.encode(batch, normalize=True)
            all_cand_embs.append(batch_embs)
        
        # 合并所有批次的 embeddings
        cand_embs = np.vstack(all_cand_embs)  # shape: (num_candidates, dim)
        
        # --- 3. 计算相似度分数 ---
        st.info("💯 Calculating the similarity score...")
        
        # 矩阵乘法计算余弦相似度
        sim_scores = np.matmul(query_emb, cand_embs.T).squeeze().tolist()
        
        # 如果只有一个候选，确保 sim_scores 是列表
        if isinstance(sim_scores, float):
            sim_scores = [sim_scores]
        
        # 按相似度排序
        scored_paths = list(zip(path_keys, sim_scores))
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top 100
        top100 = scored_paths[:100]
        logger.info(f"Top 100 paths selected from {len(path_keys)} candidates")
        
        # --- 4. 使用 Cross-encoder 重排序 ---
        st.info("🔄 Reranker fine sorting is being used...")
        
        # 构造 query-passage 对
        pairs = [(query_text, pk) for pk, _ in top100]
        
        # 分批调用 rerank API
        cross_batch_size = 16
        all_cross_scores = []
        
        for i in range(0, len(pairs), cross_batch_size):
            batch_pairs = pairs[i:i + cross_batch_size]
            st.text(f"⏳ Rerank batch {i//cross_batch_size + 1}/{(len(pairs)-1)//cross_batch_size + 1}...")
            
            # 调用 rerank API
            batch_scores = cross_api.predict(batch_pairs)
            all_cross_scores.extend(batch_scores)
        
        # --- 5. 最终排序并返回 top 30 ---
        rerank_final = list(zip([p[0] for p in top100], all_cross_scores))
        rerank_final.sort(key=lambda x: x[1], reverse=True)
        
        top30 = rerank_final[:30]
        top30_values = [path_kv[pk] for pk, _ in top30]
        
        logger.info(f"Cross-encoder reranked top 30 paths")
        st.success(f"✅ Done! Returns top {len(top30_values)}")
        
        return {"neo4j_retrieval": top30_values}
    
    except Exception as e:
        logger.error(f"Error in rerank_paths_with_apis: {str(e)}")
        st.error(f"❌ Reranking failed: {str(e)}")
        
        # 降级方案：直接返回前30个
        path_keys = list(path_kv.keys())
        fallback_values = [path_kv[k] for k in path_keys[:30]]
        return {"neo4j_retrieval": fallback_values}

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

# ======================== Prompt 模板 ========================
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
# ======================== 处理节点 ========================
def parse_query(state: MyState):
    logger.info("---NODE: parse_query---")
    user_query = [message.content for message in state["messages"] if hasattr(message, 'content')]
    query_str = user_query
    print(f"parse_query: {query_str}")
    parse_outcome = chain1.invoke({"query": query_str, "label_list": "\n".join(label_list)})
    parse_outcome_t = chain1_t.invoke({"query": query_str})
    try:
        parsed_text = getattr(parse_outcome, "content", str(parse_outcome)).strip()
        parsed_json = _extract_json_from_text(parsed_text)
        print(f"parse_json:{parsed_json}")
        entity_compound_atomic = parsed_json.get("entity", [])
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
            "interaction" : interaction,
            "parsed_query": parsed_query

        }
    except Exception as e:
        logger.warning(f"JSON failed: {e}")
        return {
            "messages": [AIMessage(content="failed to parse query")],
        }
    




def user_input(state: dict, user_reply_text = None):
    """
    Streamlit 版本：
    1. LangGraph 调用该节点时，会先返回 AI 提示语给前端。
    2. 前端显示提示语，并等待用户输入。
    3. 用户在 Streamlit 输入的内容需要由外部传入 user_reply_text。
    """
    interaction_content = state.get(
        "interaction",
        "Your question is not informative enough. Please describe the problem in more detail."
    )

    ai_message = AIMessage(content=interaction_content)

    # 情况 1：还没有收到用户输入（流程暂停，等待前端输入）
    if not user_reply_text:
        return {
            "ai_message": ai_message.content,
            "need_user_reply": True,               # 告诉前端：需要用户输入
            "messages": [],
            "user_reply": None
        }

    # 情况 2：已经收到用户输入（流程继续）
    return {
        "ai_message": ai_message.content,
        "need_user_reply": False,
        "messages": [HumanMessage(content=user_reply_text)],
        "user_reply": user_reply_text
    }



def whether_to_interact(state):
    """判断是否需要与用户交互。"""
    print("---EDGE: whether_to_interact---")
    interaction = state.get("sufficient_or_insufficient")
    print(f"interaction:{interaction}")
    if interaction == "insufficient":
        print("Decision: Insufficient information, user input required.")
        return "user_input"
    elif interaction == "sufficient":
        print("Decision: Information is sufficient to enter kg retrieval.")
        return "neo4j_retrieval"
    else:
        return "stop_flow"


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

    _, sap_api = resources["sap"]
    _, bi_api = resources["bi"]
    _, cross_api = resources["cross"]
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
            entity_embedding2 = embed_entity(parsed_query, sap_api).astype('float32').reshape(1, -1)
            D, I = idx3.search(entity_embedding2, 5)
            candidate_triples = []
            for idx in I[0]:
                idx_int = int(idx)  # ✅ 确保是 Python int
                if 0 <= idx_int < len(meta3):
                    candidate_triples.append(meta3[idx_int])
                else:
                    logger.warning(f"Index {idx_int} out of range for meta3 (len={len(meta3)})")
                    
            logger.info(f"Found {len(candidate_triples)} candidate triples")
            cand_info = [{
            "head": cand.get("head", ""),
            "head_desc": cand.get("head_desc", ""),
            "rel": cand.get("rel", ""),
            "rel_desc": cand.get("rel_desc", ""),
            "rel_id": cand.get("rel_id", ""),
            "tail": cand.get("tail", ""),
            "tail_desc": cand.get("tail_desc", "")}
            for cand in candidate_triples]
            
            entity_embedding = embed_entity(entity, sap_api).astype('float32').reshape(1, -1)
            candidates1 = []
            try:
                D1, I1 = idx1.search(entity_embedding, topk)
                for idx in I1[0]:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(meta1):
                        candidates1.append(meta1[idx_int])
                    else:
                        logger.warning(f"Index {idx_int} out of range for meta1")
                logger.info(f"idx1 returned {len(candidates1)} candidates")
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
                logger.info(f"idx2 returned {len(candidates2)} candidates")
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
            logger.warning(f"'{entity}'failed in faiss {e}")
            continue

    result = rerank_paths_with_apis(
        query_text=query_text,
        path_kv=path_kv,
        bi_api=bi_api,
        cross_api=cross_api
    )
    return result

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
    print("回答步骤")
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
    builder.add_node("user_input", user_input)
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
    builder.add_edge("user_input", "parse_query")
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
resources = load_all_resources()
graph = build_graphrag_agent(resources)
def invoke_graph_with_state(graph, state_input: dict):
    """
    调用 graph.invoke 并返回新的 state（字典）。
    state_input 可以是 {"messages": [...]} 或上一次的完整 state（并可包含 user_reply_text）
    """
    return graph.invoke(state_input)

# Streamlit UI ----------------------------------------------------
st.title("DGADIS - Streamlit Demo")

if "graph_state" not in st.session_state:
    st.session_state["graph_state"] = None   # current state returned by graph
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []  # optional: store chat turns

user_input = st.text_input("Please input your dental question:", key="initial_query")

# 初次提交用户问题
if st.button("Submit Query"):
    if not user_input or not user_input.strip():
        st.warning("Please input a question.")
    else:
        # 构造 messages 并第一次调用 graph
        inputs = {"messages": [HumanMessage(content=user_input.strip())]}
        new_state = invoke_graph_with_state(graph, inputs)
        st.session_state["graph_state"] = new_state
        # 记录用户提问（可选）
        st.session_state["conversation_history"].append(("user", user_input.strip()))
        st.rerun()

# 如果已经有 graph_state（说明流程正在进行或已完成）
state = st.session_state.get("graph_state")
if state:
    # 若节点要求补充信息（user_input 节点返回 need_user_reply True）
    if state.get("need_user_reply"):
        st.info("Agent asks:")
        st.write(state.get("ai_message", "Please provide more information."))

        # 补充信息输入框 —— 使用独立 key，避免与初始输入冲突
        reply = st.text_input("Please enter the additional info:", key="supplement_reply")

        if st.button("Continue with supplement"):
            if not reply or not reply.strip():
                st.warning("Please enter supplemental information before continuing.")
            else:
                # 将用户补充写入 state 并再次调用 graph 继续流程
                # 注意：把之前的 state 作为输入传入，同时包含 user_reply_text 字段
                # 这样 user_input 节点会接收到 user_reply_text 并返回 messages 包含 HumanMessage
                state_input = dict(state)  # shallow copy
                # 把 user_reply_text 作为临时字段注入
                state_input["user_reply_text"] = reply.strip()
                new_state = invoke_graph_with_state(graph, state_input)
                st.session_state["graph_state"] = new_state
                st.session_state["conversation_history"].append(("user", reply.strip()))
                st.rerun()

    else:
        # 如果不需要补充，查看是否有最终答案（llm_answer）
        llm_ans = state.get("llm_answer")
        if llm_ans:
            st.success("Answer from agent:")
            st.write(llm_ans)
            # 可选：显示检索到的 neo4j/knowledge results
            #if state.get("neo4j_retrieval") is not None:
                #st.subheader("Neo4j / Retrieval results")
                #st.write(state.get("neo4j_retrieval"))
            # 可选：重置会话或继续下一轮对话
            if st.button("Start new question"):
                st.session_state["graph_state"] = None
                st.session_state["conversation_history"] = []
                st.rerun()
        else:
            # 情况：既不需要补充也没有 llm_answer —— 输出当前 state 以便排查
            st.write("Current state (no further action):")
            st.json(state)
