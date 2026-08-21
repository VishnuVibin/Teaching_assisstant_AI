import os
import sys

# Restrict PyTorch / OpenMP CPU thread allocation to 1 thread to optimize memory on cloud containers (Render)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

import fitz
import faiss
import numpy as np
import requests
import re
import pickle
import gc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from generate_pdf import generate_pdf, parse_study_guide
from flask import Flask, request, jsonify
from generate_teaching_ppt import generate_slides
from generate_quiz import generate_quiz

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if API_KEY and API_KEY.startswith("tgp_"):
    API_URL = "https://api.together.xyz/v1/chat/completions"
    MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
else:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "meta-llama/llama-3.1-8b-instruct"

# Lazy Singleton pattern to ensure SentenceTransformer model is loaded only once on CPU
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model on CPU (single-threaded)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model

def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    try:
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    finally:
        doc.close()

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_embeddings(chunks, batch_size=16):
    model_instance = get_model()
    embeddings = model_instance.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")

def embed_query(question):
    model_instance = get_model()
    if isinstance(question, str):
        query_input = [question]
    else:
        query_input = question
    embeddings = model_instance.encode(
        query_input,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    if embeddings.ndim == 2 and embeddings.shape[0] == 1:
        return embeddings[0].astype("float32")
    return embeddings.astype("float32")

def clean_name(name):
    if not name:
        return ""
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'\|\s*\|', '|', name)
    return name.strip()

def clean_ocr_text(text):
    if not text:
        return ""
    replacements = {
        r'\bkometries\b': 'Isometries',
        r'\bKometries\b': 'Isometries',
        r'\bAnea\b': 'Area',
        r'\bUnes\b': 'Lines',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def is_title_case(text):
    words = [w for w in re.findall(r'\b[A-Za-z]+\b', text) if w]
    if not words:
        return False
    if not words[0][0].isupper():
        return False
    ignore_words = {'and', 'or', 'the', 'of', 'in', 'at', 'to', 'a', 'an', 'is', 'it', 'its', 'for', 'by', 'with', 'from', 'as', 'about'}
    important_words = [w for w in words if w.lower() not in ignore_words]
    if not important_words:
        return True
    upper_important = [w for w in important_words if w[0].isupper()]
    return len(upper_important) >= 0.5 * len(important_words)

def clean_title(title):
    title = re.sub(r'[\ufeff\u200b\u2003\xad]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def is_unwanted(title):
    title_lower = title.lower()
    unwanted_keywords = [
        "title page", "copyright", "contents", "table of contents", 
        "preface", "acknowledgement", "about the author", "about the editor", 
        "index", "bibliography", "glossary", "appendix", "cover", 
        "dedication", "brief contents", "contents at a glance",
        "about the technical editor", "online resources", "contributor",
        "colophon", "credits", "errata", "further reading", "author index",
        "subject index", "answers", "introduction pages", "introduction page"
    ]
    for kw in unwanted_keywords:
        if kw in title_lower:
            return True
            
    if re.search(r"^\s*part\s+(?:[ivxldcm]+|\d+)\b", title_lower):
        return True
        
    return False

def parse_indd_filename(filename):
    name_part = filename.split('.')[0].strip()
    
    num_match = re.search(r'\b(?:Unit|U|Chapter|Chap|Ch|Sec|Section)\s*[-_]?\s*(\d+)\b', name_part, re.IGNORECASE)
    if not num_match:
        num_match = re.search(r'\b(\d+)\b', name_part)
        
    num = None
    if num_match:
        num = num_match.group(1)
        
    clean_n = re.sub(r'^\s*(?:[IVXLCDM]+\s+)?(?:Unit|U|Chapter|Chap|Ch|Sec|Section)?\s*\d+\b\s*[-_\s]*', '', name_part, flags=re.IGNORECASE)
    clean_n = re.sub(r'_(?:New|new)$', '', clean_n)
    clean_n = re.sub(r'^[-_\s]+|[-_\s]+$', '', clean_n)
    
    if not clean_n:
        clean_n = name_part
        
    return num, clean_n

def detect_sections(pdf_path):
    from collections import Counter, defaultdict
    doc = fitz.open(pdf_path)
    sections = []
    try:
        try:
            toc = doc.get_toc()
        except Exception:
            toc = []

        if toc and len(toc) >= 2:
            level_counts = defaultdict(int)
            level_entries = defaultdict(list)
            
            chapter_regex = re.compile(r"\b(?:chapter|unit|ch|chap|lesson|ex\.no)\b", re.IGNORECASE)
            number_prefix_regex = re.compile(r"^\s*(?:\d+|[ivxldcm]+)[\s.:]", re.IGNORECASE)
            
            for item in toc:
                lvl = item[0]
                title = clean_name(item[1])
                level_entries[lvl].append(item)
                
                if not is_unwanted(title):
                    if chapter_regex.search(title) or number_prefix_regex.search(title):
                        level_counts[lvl] += 1
                        
            best_lvl = None
            max_count = 0
            for lvl, count in level_counts.items():
                if count > max_count:
                    max_count = count
                    best_lvl = lvl
                    
            if not best_lvl or max_count < 2:
                non_unwanted_counts = defaultdict(int)
                for item in toc:
                    lvl = item[0]
                    title = clean_name(item[1])
                    if not is_unwanted(title):
                        non_unwanted_counts[lvl] += 1
                best_lvl = 1
                max_non_unwanted = 0
                for lvl, count in non_unwanted_counts.items():
                    if count > max_non_unwanted:
                        max_non_unwanted = count
                        best_lvl = lvl
                        
            candidates = level_entries[best_lvl]
            first_chapter_found = False
            
            for idx, item in enumerate(candidates):
                title = clean_name(item[1])
                if is_unwanted(title):
                    continue
                    
                is_real_chapter = False
                if chapter_regex.search(title):
                    is_real_chapter = True
                elif number_prefix_regex.search(title):
                    is_real_chapter = True
                    
                if not first_chapter_found:
                    if is_real_chapter:
                        first_chapter_found = True
                    else:
                        continue  
                        
                start_page = item[2] - 1
                
                if idx + 1 < len(candidates):
                    end_page = candidates[idx + 1][2] - 2
                else:
                    end_page = len(doc) - 1
                    
                num_match = re.search(r"\b(?:chapter|unit|ch|chap|lesson|ex\.no\s*:?)\s*(\d+|[ivxldcm]+)\b", title, re.IGNORECASE)
                if not num_match:
                    num_match = re.search(r"^\s*(\d+|[ivxldcm]+)\b", title, re.IGNORECASE)
                    
                if num_match:
                    num = num_match.group(1).upper()
                else:
                    num = str(len(sections) + 1)
                    
                sections.append({
                    'number': num,
                    'name': clean_ocr_text(title),
                    'start_page': start_page,
                    'end_page': max(start_page, end_page)
                })

        if not sections:
            page_to_indd = {}
            indd_regex = re.compile(r'\b([a-zA-Z0-9_\s\-]+\.(?:indd|qxd))\b', re.IGNORECASE)
            
            for page_idx in range(len(doc)):
                text = doc[page_idx].get_text()
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                if not lines:
                    continue
                    
                candidate_lines = lines[:5] + lines[-5:]
                for line in candidate_lines:
                    matches = indd_regex.findall(line)
                    if matches:
                        clean_match = matches[-1].strip()
                        words = clean_match.split()
                        if len(words) > 5:
                            clean_match = " ".join(words[-5:])
                        page_to_indd[page_idx] = clean_match
                        break
                        
            if page_to_indd:
                groups = []
                current_indd = None
                start_page = None

                for page_idx in range(len(doc)):
                    indd = page_to_indd.get(page_idx)
                    if indd != current_indd:
                        if current_indd is not None:
                            groups.append({
                                'indd': current_indd,
                                'start_page': start_page,
                                'end_page': page_idx - 1
                            })
                        current_indd = indd
                        start_page = page_idx

                if current_indd is not None:
                    groups.append({
                        'indd': current_indd,
                        'start_page': start_page,
                        'end_page': len(doc) - 1
                    })

                if len(groups) >= 2:
                    for group in groups:
                        filename = group['indd']
                        num, name = parse_indd_filename(filename)
                        
                        if is_unwanted(name):
                            continue
                            
                        if name.lower() in ("chapter", "unit", "ch", "sec", "section") or re.match(r'^(?:chapter|unit|ch|sec|section)?\d+$', name, re.IGNORECASE):
                            start_text = doc[group['start_page']].get_text()
                            start_lines = [l.strip() for l in start_text.split('\n') if l.strip()]
                            for line in start_lines[:15]:
                                cleaned_line = clean_title(line)
                                if len(cleaned_line) > 5 and not cleaned_line.isdigit() and is_title_case(cleaned_line):
                                    if not any(w in cleaned_line.lower() for w in ("learning objectives", "introduction", "summary", "exercise")):
                                        name = cleaned_line
                                        break
                                        
                        if not num:
                            num = str(len(sections) + 1)
                            
                        sections.append({
                            'number': num,
                            'name': clean_ocr_text(clean_name(name)),
                            'start_page': group['start_page'],
                            'end_page': group['end_page']
                        })

        if not sections:
            raw_sections = []
            toc_pages = set()
            appendix_pages = set()
            
            for page_idx in range(len(doc)):
                text = doc[page_idx].get_text()
                first_lines = [l.strip().lower() for l in text.split('\n') if l.strip()][:10]
                if any(w in first_lines for w in ["contents", "table of contents", "t.o.c."]):
                    toc_pages.add(page_idx)
                if any(w in text for w in ["ANSWERS TO SELECTED EXERCISES", "Answers to Selected Exercises", "BIBLIOGRAPHY"]):
                    appendix_pages.add(page_idx)
                    
            seen_numbers = set()
            
            for page_idx in range(len(doc)):
                if page_idx in toc_pages:
                    continue
                    
                text = doc[page_idx].get_text()
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                if not lines:
                    continue
                    
                found = False
                for idx, line in enumerate(lines[:15]):
                    if len(line) > 60:
                        continue
                        
                    match1 = re.match(r'^(?:Chapter|CHAPTER|chap|CHAP|Section|SECTION|Unit|UNIT)\s+([IVXLCDM]+|\d+)\b(.*)', line, re.IGNORECASE)
                    match2 = re.match(r'^(\d+)\s+([A-Za-z][A-Za-z\s&,;:\-\xad]+)$', line)
                    
                    num = None
                    name = None
                    
                    if match1:
                        num = match1.group(1).lower().strip()
                        rest = match1.group(2).strip()
                        if not rest and idx + 1 < len(lines) and len(lines[idx+1]) < 60:
                            rest = lines[idx+1].strip()
                        name = f"Chapter {num.upper()}" + (f" - {rest}" if rest else "")
                    elif match2:
                        num_candidate = match2.group(1).strip()
                        title = match2.group(2).strip()
                        if is_title_case(title):
                            num = num_candidate
                            name = f"Chapter {num} - {title}"
                    elif line.upper() in ("UNIT", "CHAPTER") and idx + 1 < len(lines) and lines[idx+1].strip().isdigit():
                        num = lines[idx+1].strip()
                        title_candidates = []
                        search_range = lines[max(0, idx-5):idx] + lines[idx+2:min(len(lines), idx+7)]
                        for candidate in search_range:
                            cleaned_c = clean_title(candidate)
                            if len(cleaned_c) > 5 and not cleaned_c.isdigit() and is_title_case(cleaned_c):
                                if not any(w in cleaned_c.lower() for w in ("learning objectives", "introduction", "summary", "exercise", "unit", "chapter")):
                                    title_candidates.append(cleaned_c)
                        title = title_candidates[0] if title_candidates else f"Unit {num}"
                        name = f"Chapter {num} - {title}"
                        
                    if num and name:
                        if is_unwanted(name):
                            continue
                        if num.isdigit():
                            num = str(int(num))
                        else:
                            num = num.lower()
                            
                        if num.startswith('7') and len(num) == 2 and num != '7':
                            corrected_num = '1' + num[1]
                            num = corrected_num
                            if name:
                                name = name.replace(f"Chapter 7", f"Chapter 1")
                        
                        if num not in seen_numbers:
                            if page_idx in appendix_pages:
                                if "answers" not in name.lower() and "selected exercises" not in name.lower():
                                    continue
                                    
                            seen_numbers.add(num)
                            raw_sections.append({
                                'number': num,
                                'name': clean_ocr_text(clean_name(name)),
                                'start_page': page_idx
                            })
                            found = True
                            break
                if found:
                    continue

            raw_sections = sorted(raw_sections, key=lambda x: x['start_page'])
            
            for idx, sec in enumerate(raw_sections):
                start = sec['start_page']
                if idx + 1 < len(raw_sections):
                    end = raw_sections[idx+1]['start_page'] - 1
                else:
                    end = len(doc) - 1
                sections.append({
                    'number': sec['number'],
                    'name': sec['name'],
                    'start_page': start,
                    'end_page': max(start, end)
                })
    finally:
        doc.close()
    return sections

def find_matching_section(query, sections):
    query_clean = query.strip().lower()
    
    def normalize_num(val):
        val = val.lower().strip()
        if val.isdigit():
            return str(int(val))
        return val
        
    norm_query = normalize_num(query_clean)
    
    for sec in sections:
        if normalize_num(sec['number']) == norm_query:
            return sec
            
    num_match = re.search(r"\b(\d+|[ivxldcm]+)\b", query_clean)
    if num_match:
        target_num = normalize_num(num_match.group(1))
        for sec in sections:
            if normalize_num(sec['number']) == target_num:
                return sec
                
    best_match = None
    best_score = 0
    for sec in sections:
        sec_name_clean = sec['name'].lower()
        if query_clean in sec_name_clean:
            score = len(query_clean) / len(sec_name_clean)
            if score > best_score:
                best_score = score
                best_match = sec
                
    if best_match:
        return best_match

    if len(query_clean) >= 4:
        for sec in sections:
            sec_name_clean = sec['name'].lower()
            words = [w for w in query_clean.split() if len(w) >= 4]
            for w in words:
                if w in sec_name_clean:
                    score = len(w) / len(sec_name_clean)
                    if score > best_score:
                        best_score = score
                        best_match = sec
                        
    return best_match

class Retrieval:
    def __init__(self, pdf_path, vectordb, embed_query_fn, sections=None):
        self.pdf_path = pdf_path
        self.db = vectordb
        self.embed_query = embed_query_fn
        if sections is not None:
            self.sections = sections
        else:
            print("Analyzing PDF structure and detecting chapters...")
            self.sections = detect_sections(pdf_path)

    def retrieve(self, question):
        matched_sec = find_matching_section(question, self.sections)
        
        if matched_sec:
            print(f"\n[System] Found matching chapter: {matched_sec['name']} (Pages {matched_sec['start_page']+1}-{matched_sec['end_page']+1})")
            doc = fitz.open(self.pdf_path)
            try:
                context = ""
                for page_idx in range(matched_sec['start_page'], min(matched_sec['end_page'] + 1, len(doc))):
                    context += doc[page_idx].get_text()
                return context, matched_sec
            finally:
                doc.close()
            
        print("\n[System] Section not matched. Falling back to semantic search...")
        query_embedding = self.embed_query(question)
        semantic_context = "\n\n".join(self.db.search(query_embedding, k=4))
        return semantic_context, None

FAISS_INDEX_FILE = "faiss_index.bin"
INDEX_META_FILE = "index_meta.pkl"

class VectorDB:
    def __init__(self):
        self.index = None
        self.documents = []

    def build(self, chunks, embeddings):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.documents = chunks

    def save(self, index_path=FAISS_INDEX_FILE, docs_path=INDEX_META_FILE, extra_meta=None):
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            meta = {
                "documents": self.documents,
                "extra": extra_meta or {}
            }
            with open(docs_path, "wb") as f:
                pickle.dump(meta, f)

    def load(self, index_path=FAISS_INDEX_FILE, docs_path=INDEX_META_FILE):
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                meta = pickle.load(f)
            if isinstance(meta, dict) and "documents" in meta:
                self.documents = meta["documents"]
                extra = meta.get("extra", {})
            else:
                self.documents = meta
                extra = {}
            return extra
        return None

    def search(self, query_embedding, k=3):
        if self.index is None or len(self.documents) == 0:
            return []
        query_arr = np.array(query_embedding).astype("float32")
        if query_arr.ndim == 1:
            query_arr = np.expand_dims(query_arr, axis=0)
        distance, index = self.index.search(query_arr, k)
        results = []
        for i in index[0]:
            if 0 <= i < len(self.documents):
                results.append(self.documents[i])
        return results

def generate_answer(question, context, matched_sec=None):
    if matched_sec:
        section_num = matched_sec['number']
        section_name = matched_sec['name']
    else:
        section_num = "Not Available"
        section_name = "Not Available"

    prompt = f"""
You are an expert AI Teaching Assistant and Career Mentor.

Your task is to analyze the retrieved context from the textbook/document and generate a comprehensive, highly optimized study guide for the specified chapter or section.

The guide must be extremely detailed, complete, and require no further regeneration or modifications. Output the guide using the exact Markdown format shown below.

Context to analyze:
{context}

---
Output Format:

Chapter Number: {section_num}
Chapter Name: {section_name}

------------------------------------
Chapter Syllabus (Topics Covered)
------------------------------------
Provide a clean bulleted list of all main topics and subtopics discussed in this section.
• Topic 1
• Topic 2
• ...

------------------------------------
Important Topics to Master
------------------------------------
For each major topic or concept, explain in detail:
• Topic Name
  - Why this topic is important: Provide a clear technical or academic explanation of its importance.
  - Important concepts to understand: Explain the core concepts, theories, step-by-step logic, structures, and functions in depth as discussed in the text.

------------------------------------
Career-Oriented Topics
------------------------------------
Identify and list the top topics that are most important from a career, industry, and placement perspective.
- Arrange them in order of priority.
- Do not add descriptions or bullet points under these topics—return ONLY their names.
• Topic Name 1
• Topic Name 2
• ...

------------------------------------
Chapter Summary
------------------------------------
Provide a cohesive, comprehensive summary of the entire chapter covering all major points and transitions.
• Point 1
• Point 2
• Point 3
• ...

------------------------------------
Key Takeaways
------------------------------------
Provide the ultimate cheat sheet for this section:
• Important definitions: Give precise definitions of key terms.
• Important formulas / syntax: (If applicable) List any mathematical formulas, command-line examples, or programming syntax.
• Important concepts: Summarize the core underlying principles.
• Frequently asked interview topics: List 3-5 specific questions a student might be asked in an interview about this chapter, along with brief hints.
• Exam-focused points: List the highly testable concepts.

------------------------------------
Teaching Plan (Instructor Guide)
------------------------------------
Provide a highly structured, step-by-step teaching plan to help the instructor teach this chapter/section perfectly. It should include:
• Day-by-Day Breakdown: Recommended sequencing of topics with timelines styled as Day 1, Day 2, Day 3, etc.
• Learning Objectives: What the students should be able to do/explain after each day.
• Teaching Methods & Strategies: Suggested instructional approaches (e.g., analogical explanations, hands-on demos).
• Classroom Activities & Discussion Prompts: Active learning exercises to engage students.
• Check-on-Learning Questions: In-class questions to gauge student understanding in real-time.

User Question/Context Query:
{question}
"""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI Teaching Assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )
    
    try:
        result = response.json()
    except Exception as e:
        print("Failed to parse API response:", e)
        return "Error generating response."

    if "choices" not in result:
        print("API Error Response:", result)
        return "Error generating response."
        
    return result["choices"][0]["message"]["content"]

app = Flask(__name__)

retriever = None

def init_retriever_from_cache():
    global retriever
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(INDEX_META_FILE):
        try:
            db = VectorDB()
            extra_meta = db.load(FAISS_INDEX_FILE, INDEX_META_FILE)
            if extra_meta and "pdf_path" in extra_meta and os.path.exists(extra_meta["pdf_path"]):
                sections = extra_meta.get("sections")
                retriever = Retrieval(extra_meta["pdf_path"], db, embed_query, sections=sections)
                print(f"[System] Successfully auto-loaded cached vector index for '{extra_meta['pdf_path']}' on startup.")
        except Exception as e:
            print(f"[System] Warning: Could not auto-load cached index: {e}")

init_retriever_from_cache()

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global retriever

    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "JSON body is required"
        }), 400

    pdf_path = data.get("pdf_path")

    if not pdf_path:
        return jsonify({
            "status": "error",
            "message": "PDF path is required"
        }), 400

    if not os.path.exists(pdf_path):
        return jsonify({
            "status": "error",
            "message": f"File '{pdf_path}' not found"
        }), 404

    try:
        db = VectorDB()
        extra_meta = db.load(FAISS_INDEX_FILE, INDEX_META_FILE)
        
        if extra_meta is not None and extra_meta.get("pdf_path") == pdf_path and os.path.exists(FAISS_INDEX_FILE):
            print(f"\n[System] Loading pre-built FAISS index and sections for '{pdf_path}' from disk...")
            sections = extra_meta.get("sections")
            retriever = Retrieval(pdf_path, db, embed_query, sections=sections)
        else:
            print("\nExtracting PDF text...")
            text = extract_pdf(pdf_path)

            print("Chunking and indexing text...")
            chunks = chunk_text(text)
            embeddings = create_embeddings(chunks, batch_size=16)

            print("Building Vector Database...")
            db.build(chunks, embeddings)

            sections = detect_sections(pdf_path)
            db.save(FAISS_INDEX_FILE, INDEX_META_FILE, extra_meta={"pdf_path": pdf_path, "sections": sections})
            
            retriever = Retrieval(pdf_path, db, embed_query, sections=sections)
            gc.collect()

        chapters = []
        if retriever.sections:
            print("\n====================================")
            print("Detected Chapters / Sections")
            print("====================================")
            for s in retriever.sections:
                print(f"[{s['number']}] {s['name']} (Pages {s['start_page']+1}-{s['end_page']+1})")
                chapters.append({
                    "number": s["number"],
                    "name": s["name"],
                    "start_page": s["start_page"] + 1,
                    "end_page": s["end_page"] + 1
                })
            print("====================================")
        else:
            print("No chapter structure detected.")

        return jsonify({
            "status": "success",
            "message": "PDF uploaded successfully",
            "chapters": chapters
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    global retriever

    if retriever is None:
        return jsonify({
            "status": "error",
            "message": "Upload a PDF first."
        }), 400

    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "JSON body is required."
        }), 400

    question = data.get("question")

    if not question:
        return jsonify({
            "status": "error",
            "message": "Question is required."
        }), 400

    try:
        context, matched_sec = retriever.retrieve(question)

        print("\nGenerating study guide / answering query...")
        answer = generate_answer(question, context, matched_sec)

        pdf_filename = generate_pdf(answer, question, matched_sec)

        ppt_name = None
        quiz_name = None

        if matched_sec:
            safe_name = re.sub(
                r'[\\/:*?"<>|]',
                "_",
                matched_sec["name"]
            )
            ch_name = matched_sec["name"]
            ch_num = matched_sec["number"]
            ppt_name = f"Chapter_{ch_num}_{safe_name}.pptx"

            generate_slides(
                context=context,
                chapter_name=ch_name,
                chapter_number=ch_num,
                output_ppt_name=ppt_name
            )

            quiz_name = f"Quiz_Chapter_{ch_num}.json"

            generate_quiz(
                context=context,
                chapter_name=ch_name,
                chapter_number=ch_num,
                output_file=quiz_name
            )
        else:
            safe_name = re.sub(
                r'[\\/:*?"<>|]',
                "_",
                question[:30]
            )
            ppt_name = f"General_Q_and_A_{safe_name}.pptx"

            generate_slides(
                context=context,
                chapter_name=question,
                chapter_number="N/A",
                output_ppt_name=ppt_name
            )

        parsed_sections = parse_study_guide(answer)

        syllabus = ""
        important = ""
        career = ""
        summary = ""
        key_takeaway = ""
        teaching_plan = ""

        for sec in parsed_sections:
            title_lower = sec['title'].lower()
            if "syllabus" in title_lower or "topics covered" in title_lower:
                syllabus = sec['content']
            elif "important topics" in title_lower:
                important = sec['content']
            elif "career" in title_lower:
                career = sec['content']
            elif "summary" in title_lower:
                summary = sec['content']
            elif "key takeaway" in title_lower:
                key_takeaway = sec['content']
            elif "teaching plan" in title_lower:
                teaching_plan = sec['content']

        pdf_name = pdf_filename
        gc.collect()

        return jsonify({
            "status": "success",
            "syllabus": syllabus,
            "important_topics": important,
            "career_topics": career,
            "chapter_summary": summary,
            "pdf_file": pdf_name,
            "ppt_file": ppt_name,
            "quiz_file": quiz_name,
            "keytakeaway": key_takeaway,
            "teaching": teaching_plan
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)