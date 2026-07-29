import requests
import json
from dotenv import load_dotenv
import os
# ----------------------------
# OpenRouter Configuration
# ----------------------------
load_dotenv()
API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-3.1-8b-instruct"
URL="https://openrouter.ai/api/v1/chat/completions"


def generate_quiz(
    context,
    chapter_name,
    chapter_number,
    output_file="quiz.json"
):
    """
    Generates a 10-question MCQ quiz from the retrieved chapter.
    """

    prompt = f"""
You are an expert educator and assessment designer.

Generate exactly 10 multiple-choice questions using ONLY the provided chapter.

Chapter Number:
{chapter_number}

Chapter Name:
{chapter_name}

Chapter Content:
{context}

Rules:

1. Generate exactly 10 different types of questions.

2. Every question must have four options:
A
B
C
D

3. Only one option must be correct.

4. Wrong answers should be realistic.

5. Cover different topics from the chapter.

6. Mix difficulty:
- Easy
- Medium
- Hard

7. Do NOT invent information.

8. Return ONLY valid JSON.

Return format:

{{
    "chapter_number":"{chapter_number}",
    "chapter_name":"{chapter_name}",
    "questions":[
        {{
            "question":"Question text",
            "options": {{
                "A":"Option A",
                "B":"Option B",
                "C":"Option C",
                "D":"Option D"
            }},
            "answer":"A",
            "difficulty":"Easy",
            "explanation":"Why A is correct."
        }}
    ]
}}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }

    print(f"Generating Quiz using {MODEL}...")

    try:

        response = requests.post(
            URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Remove markdown if present
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        quiz = json.loads(content)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(quiz, f, indent=4, ensure_ascii=False)

        print(f"\nQuiz saved as {output_file}")

        return quiz

    except Exception as e:
        print("Quiz Generation Error:", e)
        return None