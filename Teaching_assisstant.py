import fitz
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import re
import os
import sys
from dotenv import load_dotenv
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from generate_teaching_ppt import generate_slides

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-3.1-8b-instruct"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_embeddings(chunks):
    return np.array(model.encode(chunks)).astype("float32")

def embed_query(question):
    return np.array(model.encode(question)).astype("float32")

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

def detect_sections(pdf_path):
    from collections import Counter
    doc = fitz.open(pdf_path)
    sections = []

    try:
        toc = doc.get_toc()
    except Exception:
        toc = []

    if toc and len(toc) >= 2:
        lvl1 = [item for item in toc if item[0] == 1]
        if len(lvl1) >= 2:
            for idx, item in enumerate(lvl1):
                title = clean_name(item[1])
                start_page = item[2] - 1  

                if idx + 1 < len(lvl1):
                    end_page = lvl1[idx + 1][2] - 2
                else:
                    end_page = len(doc) - 1
                
                num_match = re.search(r"^(?:chapter|unit|section|ex\.no\s*:)?\s*(\d+)\b", title, re.IGNORECASE)
                num = str(idx + 1)
                if num_match:
                    num = num_match.group(1)
                
                sections.append({
                    'number': num,
                    'name': clean_ocr_text(title),
                    'start_page': start_page,
                    'end_page': max(start_page, end_page)
                })

    if not sections:
        page_to_indd = {}
        for page_idx in range(len(doc)):
            text = doc[page_idx].get_text()
            matches = re.findall(r'(\b[\w\-]+(?:unit|chapter|chap|ch|sec|section)[\w\-]*\.(?:indd|qxd)\b)', text, re.IGNORECASE)
            if matches:
                page_to_indd[page_idx] = matches[0].strip()
                
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
                    match = re.search(r'\d+_([A-Za-z]+)_Unit_(\d+)', group['indd'], re.IGNORECASE)
                    if match:
                        subject = match.group(1).capitalize()
                        unit_num = match.group(2)
                        num = f"{subject} Unit {unit_num}"
                    else:
                        match_generic = re.search(r'(?:unit|chapter|chap|ch|sec|section)_?(\d+)', group['indd'], re.IGNORECASE)
                        if match_generic:
                            num = match_generic.group(1)
                        else:
                            num = group['indd'].split('.')[0]
                    
                    candidates = []
                    for page_idx in range(group['start_page'], group['end_page'] + 1):
                        text = doc[page_idx].get_text()
                        lines = [l.strip() for l in text.split('\n') if l.strip()]
                        for line in lines[:3]:
                            cleaned = clean_title(line)
                            if cleaned.isdigit() or len(cleaned) < 5:
                                continue
                            if cleaned.lower() in ("exercise", "exercises", "introduction", "learning objectives", "summary", "activity", "toc", "contents"):
                                continue
                            candidates.append(cleaned)
                    
                    if candidates:
                        title = Counter(candidates).most_common(1)[0][0]
                    else:
                        title = num
                        
                    sections.append({
                        'number': num,
                        'name': clean_ocr_text(clean_name(title)),
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
            for idx, line in enumerate(lines[:3]):
                if len(line) > 60:
                    continue
                    
                match1 = re.match(r'^(?:Chapter|CHAPTER|chap|CHAP|Section|SECTION|Unit|UNIT|Exercise|EXERCISE|Ex|EX)\s+([IVXLCDM]+|\d+)\b(.*)', line)
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
                    
                if num and name:
                    if num.isdigit():
                        num = str(int(num))
                    else:
                        num = num.lower()
                        
                    # OCR Correction for numbers
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
    def __init__(self, pdf_path, vectordb, embed_query):
        self.pdf_path = pdf_path
        self.db = vectordb
        self.embed_query = embed_query
        print("Analyzing PDF structure and detecting chapters...")
        self.sections = detect_sections(pdf_path)

    def retrieve(self, question):
        matched_sec = find_matching_section(question, self.sections)
        
        if matched_sec:
            print(f"\n[System] Found matching chapter: {matched_sec['name']} (Pages {matched_sec['start_page']+1}-{matched_sec['end_page']+1})")
            doc = fitz.open(self.pdf_path)
            context = ""
            for page_idx in range(matched_sec['start_page'], matched_sec['end_page'] + 1):
                context += doc[page_idx].get_text()
            doc.close()
            return context, matched_sec
            
        print("\n[System] Section not matched. Falling back to semantic search...")
        query_embedding = self.embed_query(question)
        semantic_context = "\n\n".join(self.db.search(query_embedding, k=4))
        return semantic_context, None
class VectorDB:
    def __init__(self):
        self.index = None
        self.documents = []

    def build(self, chunks, embeddings):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.documents = chunks

    def search(self, query_embedding, k=3):
        distance, index = self.index.search(np.array([query_embedding]), k)
        results = []
        for i in index[0]:
            if i < len(self.documents):
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
        "https://openrouter.ai/api/v1/chat/completions",
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



def markdown_to_html(text):
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', lambda m: f"<b>{m.group(1) or m.group(2) or ''}</b>", text)
    text = re.sub(r'\*(.*?)\*|_(.*?)_', lambda m: f"<i>{m.group(1) or m.group(2) or ''}</i>", text)
    return text

def parse_study_guide(text):
    target_headings = [
        ("Chapter Syllabus (Topics Covered)", ["chapter syllabus", "topics covered"]),
        ("Important Topics to Master", ["important topics to master", "important topics"]),
        ("Career-Oriented Topics", ["career-oriented topics", "career oriented topics", "careeroriented topics"]),
        ("Chapter Summary", ["chapter summary", "summary"]),
        ("Key Takeaways", ["key takeaways", "takeaways"]),
        ("Teaching Plan (Instructor Guide)", ["teaching plan", "instructor guide"])
    ]
    
    lines = text.split('\n')
    sections = []
    
    current_title = "Header"
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        clean_line = re.sub(r'[*#_\-]', '', stripped).strip().lower()
        matched_heading = None
        for display_title, patterns in target_headings:
            if any(p in clean_line for p in patterns):
                if len(clean_line) < 60:
                    matched_heading = display_title
                    break
        
        if matched_heading:
            if current_content or current_title != "Header":
                clean_content_lines = []
                for l in current_content:
                    l_strip = l.strip()
                    if l_strip and (all(c == '-' for c in l_strip) or all(c == '=' for c in l_strip)) and len(l_strip) >= 5:
                        continue
                    clean_content_lines.append(l)
                
                sections.append({
                    'title': current_title,
                    'content': '\n'.join(clean_content_lines).strip()
                })
            current_title = matched_heading
            current_content = []
        else:
            current_content.append(line)
            
    if current_content or current_title != "Header":
        clean_content_lines = []
        for l in current_content:
            l_strip = l.strip()
            if l_strip and (all(c == '-' for c in l_strip) or all(c == '=' for c in l_strip)) and len(l_strip) >= 5:
                continue
            clean_content_lines.append(l)
        sections.append({
            'title': current_title,
            'content': '\n'.join(clean_content_lines).strip()
        })
        
    return sections


def draw_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#718096'))
    canvas.setStrokeColor(colors.HexColor('#E2E8F0'))
    canvas.setLineWidth(0.5)
    canvas.line(54, 55, doc.width + 54, 55)
    canvas.drawString(54, 40, "Generated by AI Teaching Assistant")
    canvas.drawRightString(doc.width + 54, 40, f"Page {doc.page}")
    canvas.restoreState()

def draw_header_footer(canvas, doc):
    draw_page_number(canvas, doc)
    canvas.saveState()
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(colors.HexColor('#4F46E5'))
    canvas.drawString(54, 750, "AI TEACHING ASSISTANT - TEACHING PLAN & GUIDE")
    canvas.setStrokeColor(colors.HexColor('#E2E8F0'))
    canvas.setLineWidth(0.5)
    canvas.line(54, 740, doc.width + 54, 740)
    canvas.restoreState()

def create_horizontal_rule():
    t = Table([['']], colWidths=['100%'], rowHeights=[2])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#E2E8F0')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    return t

def create_section_heading(title, styles):
    p = Paragraph(f"<b>{title}</b>", ParagraphStyle(
        'SideHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1A365D'),
        keepWithNext=True
    ))
    t = Table([[p]], colWidths=['100%'])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('LINELEFT', (0,0), (0,-1), 4, colors.HexColor('#4F46E5')),
    ]))
    return t

def process_section_lines(content_text, styles):
    flowables = []
    lines = content_text.split('\n')
    current_paragraph = []
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    nested_bullet_style = ParagraphStyle(
        'CustomNestedBullet',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#4A5568'),
        leftIndent=35,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    def flush_paragraph():
        if current_paragraph:
            text = ' '.join(current_paragraph).strip()
            text = re.sub(r'\s+', ' ', text)
            if text:
                html_text = markdown_to_html(text)
                flowables.append(Paragraph(html_text, body_style))
            current_paragraph.clear()
            
    prefixes = ('• ', '* ', '- ', '•\t', '*\t', '-\t')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            if flowables and not isinstance(flowables[-1], Spacer):
                flowables.append(Spacer(1, 6))
            continue
            
        if stripped.startswith(prefixes) or stripped in ('•', '*', '-'):
            flush_paragraph()
            has_leading_space = len(line) - len(line.lstrip()) > 0
            bullet_content = stripped[2:] if stripped.startswith(prefixes) else ''
            html_text = markdown_to_html(bullet_content.strip())
            
            if has_leading_space:
                flowables.append(Paragraph(f"&bull; {html_text}", nested_bullet_style))
            else:
                flowables.append(Paragraph(f"&bull; {html_text}", bullet_style))
        else:
            current_paragraph.append(line)
            
    flush_paragraph()
    return flowables

def generate_pdf(answer, question, matched_sec=None):
    parsed_sections = parse_study_guide(answer)
    
    chapter_num = "Not Available"
    chapter_name = "Not Available"
    
    header_sec = next((s for s in parsed_sections if s['title'] == 'Header'), None)
    if header_sec:
        for line in header_sec['content'].split('\n'):
            if line.lower().startswith("chapter number:"):
                chapter_num = line.split(":", 1)[1].strip()
            elif line.lower().startswith("chapter name:"):
                chapter_name = line.split(":", 1)[1].strip()
                
    if matched_sec:
        if chapter_num == "Not Available":
            chapter_num = matched_sec.get('number', 'Not Available')
        if chapter_name == "Not Available":
            chapter_name = matched_sec.get('name', 'Not Available')
            
    if chapter_num == "Not Available" and chapter_name == "Not Available":
        chapter_name = question
        
    pdf_filename = "teaching_plan.pdf"
    
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    title_top = Paragraph("AI TEACHING ASSISTANT", ParagraphStyle(
        'TitleTop',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#4F46E5'),
        spaceAfter=4
    ))
    story.append(title_top)
    
    title_text = f"Chapter {chapter_num}: {chapter_name}" if chapter_num != "Not Available" else chapter_name
    title_main = Paragraph(title_text, ParagraphStyle(
        'TitleMain',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=10
    ))
    story.append(title_main)
    story.append(create_horizontal_rule())
    story.append(Spacer(1, 15))
    
    for section in parsed_sections:
        if section['title'] == 'Header' or not section['title'].strip():
            continue
            
        story.append(create_section_heading(section['title'], styles))
        story.append(Spacer(1, 6))
        
        section_flowables = process_section_lines(section['content'], styles)
        story.extend(section_flowables)
        story.append(Spacer(1, 12))
        
    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_header_footer)
    return pdf_filename


if __name__ == "__main__":
    while True:
        pdf_path = input("Enter PDF Path: ").strip()
        if os.path.exists(pdf_path):
            break
        print(f"Error: File '{pdf_path}' not found. Please try again.")

    print("\nExtracting PDF text...")
    text = extract_pdf(pdf_path)
    
    print("Chunking and indexing text...")
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)

    print("Building Vector Database...")
    db = VectorDB()
    db.build(chunks, embeddings)

    retriever = Retrieval(pdf_path, db, embed_query)

    if retriever.sections:
        print("\n====================================")
        print("Detected Chapters / Sections in PDF:")
        print("====================================")
        for s in retriever.sections:
            print(f"[{s['number']}] {s['name']} (Pages {s['start_page']+1}-{s['end_page']+1})")
        print("====================================\n")
    else:
        print("\n[Warning] No chapters or section structures could be auto-detected in the PDF layout. Defaulting to general Q&A mode.\n")

    while True:
        question = input("Enter Chapter Name, Number, or Question (or 'exit' to quit): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        context, matched_sec = retriever.retrieve(question)
        
        print("\nGenerating study guide / answering query...")
        answer = generate_answer(question, context, matched_sec)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "="*60 + "\n")

        try:
            pdf_filename = generate_pdf(answer, question, matched_sec)
            print(f"[System] Teaching plan PDF generated successfully: {pdf_filename}")
            if matched_sec:
                safe_name = re.sub(r'[\\/:*?"<>|]', "_", matched_sec["name"])
                ch_name = matched_sec["name"]
                ch_num = matched_sec["number"]
                ppt_name = f"Chapter_{ch_num}_{safe_name}.pptx"
            else:
                safe_name = re.sub(r'[\\/:*?"<>|]', "_", question[:30])
                ch_name = question
                ch_num = "N/A"
                ppt_name = f"General_Q_and_A_{safe_name}.pptx"
                
            generate_slides(
                context=context,
                chapter_name=ch_name,
                chapter_number=ch_num,
                output_ppt_name=ppt_name)

            print("[System] PPT generated successfully!")
        except Exception as e:
            print(f"[Error] Failed to generate PDF: {e}")

