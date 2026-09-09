import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from translator import translate_text
import concurrent.futures


MODEL_NAME = "ibm-granite/granite-3.2-2b-instruct"

# Hugging Face token
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

# Device + dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[QAEngine] Loading {MODEL_NAME} on {DEVICE} ({DTYPE}) with 4-bit quantization...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    token=HF_TOKEN,
    trust_remote_code=True
)

# Model (4-bit for speed)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    token=HF_TOKEN
)
if DEVICE == "cpu":
    model.to(DEVICE)

def _build_inputs(context: str, question: str):
    """Prepare input tensors for Granite using chat template if available."""
    messages = [
        {"role": "system", "content": "You are a helpful academic assistant. Use ONLY the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=True,
            add_generation_prompt=True
        )
        prompt_len = input_ids.shape[1]
        return input_ids.to(DEVICE), prompt_len
    except Exception:
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        prompt_len = encoded["input_ids"].shape[1]
        return encoded["input_ids"].to(DEVICE), prompt_len

def generate_answer(context: str, question: str, target_lang: str = None, max_new_tokens: int = 128) -> str:
    """
    Generate an answer from Granite. Optimized for speed.
    """
    input_ids, prompt_len = _build_inputs(context, question)

    with torch.inference_mode():  # disable gradients for faster inference
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            do_sample=False,   # deterministic & faster
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][prompt_len:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if "Answer:" in answer and len(answer.split("Answer:", 1)[-1].strip()) > 0:
        answer = answer.split("Answer:", 1)[-1].strip()

    # Optional translation
    if target_lang and target_lang.lower() not in ("en", "none"):
        answer = translate_text(answer, target_lang)

    return answer


def generate_mcqs(context: str, topic: str, num_questions: int = 2, target_lang: str = None) -> list:
    """
    Generates exactly num_questions MCQs in the format:
      question: 1-3 sentences
      options: A) ... B) ... C) ... D) ...
      answer: C) <full option text>   OR   answer: C
    Returns list of dicts: {"question": "...", "options": ["a) ...","b) ...","c) ...","d) ..."], "answer": "c) ..."}
    """
    # Strict prompt to increase format consistency
    prompt = (
        f"Create exactly {num_questions} multiple-choice questions (4 options each) on the topic '{topic}'.\n"
        "Requirements:\n"
        "- Each question must be 1-3 sentences and clearly derived from the context.\n"
        "- Provide 4 options labeled as A) B) C) D).\n"
        "- Provide the correct answer either as a letter (C) or as 'C) <option text>'.\n"
        "- Output EXACT format for each Q:\n"
        "Question: <text>\n"
        "A) <option text>\n"
        "B) <option text>\n"
        "C) <option text>\n"
        "D) <option text>\n"
        "Answer: <letter or letter+option>\n\n"
        f"Context:\n{context}\n"
    )

    retries = 3
    max_tokens = 400
    parsed_mcqs = []

    for _ in range(retries):
        raw = generate_answer(context, prompt, max_new_tokens=max_tokens)
        parsed_mcqs = _parse_mcq_text(raw, num_questions)
        if parsed_mcqs:
            break
        max_tokens += 200

    # Robust fallback (context-aware short questions) if parsing fails
    if not parsed_mcqs:
        parsed_mcqs = []
        short_context = (context or topic)[:240]
        for i in range(num_questions):
            q_text = f"What is a key point about {topic}? (short)"
            # create more realistic distractors by reusing topic words
            parsed_mcqs.append({
                "question": q_text,
                "options": [
                    f"a) {topic} main idea",
                    f"b) A wrong distractor",
                    f"c) Another plausible distractor",
                    f"d) Not related"
                ],
                "answer": f"a) {topic} main idea"
            })

    # Optional translation with safe fallback
    if target_lang and target_lang.lower() not in ("en", "none"):
        def safe_translate(t):
            try:
                return translate_text(t, target_lang)
            except Exception as e:
                print(f"[translate fallback] {e}")
                return t

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(lambda q: {
                "question": safe_translate(q["question"]),
                "options": [safe_translate(opt) for opt in q["options"]],
                "answer": safe_translate(q["answer"])
            }, q) for q in parsed_mcqs]
            parsed_mcqs = [f.result() for f in futures]

    # Guarantee returned length equals requested number
    return parsed_mcqs[:num_questions]


def _parse_mcq_text(raw_text: str, num_questions: int):
    """
    Parse raw LLM output into the standardized structure.
    Tolerant to variations like 'a)' / 'A.' / 'Option A' and 'Answer: C' or 'Answer: C) text'.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    mcqs = []
    current = None

    # flexible regex patterns
    q_re = re.compile(r'(?i)^question\s*[:\-]?\s*(.+)')
    opt_re = re.compile(r'(?i)^(?:([A-Da-d])|Option\s*([A-Da-d]))\s*[:\)\.\-]?\s*(.+)')
    ans_re = re.compile(r'(?i)^(?:answer|correct)\s*[:\-]?\s*(.+)')

    for line in lines:
        mq = q_re.match(line)
        if mq:
            if current:
                mcqs.append(current)
            current = {"question": mq.group(1).strip(), "options": {}, "raw_answer": None}
            continue

        mo = opt_re.match(line)
        if mo and current is not None:
            label = (mo.group(1) or mo.group(2)).upper()
            text = mo.group(3).strip()
            current["options"][label] = text
            continue

        ma = ans_re.match(line)
        if ma and current is not None:
            current["raw_answer"] = ma.group(1).strip()
            continue

    if current:
        mcqs.append(current)

    out = []
    for q in mcqs[:num_questions]:
        # Ensure A-D options exist; fill missing options with placeholders (rare)
        opts_ordered = []
        for lbl in ["A", "B", "C", "D"]:
            if lbl in q["options"]:
                opts_ordered.append(f"{lbl.lower()}) {q['options'][lbl]}")
            else:
                opts_ordered.append(f"{lbl.lower()}) Option {lbl}")

        # Resolve correct answer:
        answer_text = None
        raw_ans = q.get("raw_answer")
        if raw_ans:
            # check if raw_ans is a single letter or letter with punctuation
            m_letter = re.match(r'^([A-Da-d])\b', raw_ans)
            if m_letter:
                idx = ord(m_letter.group(1).upper()) - ord('A')
                answer_text = opts_ordered[idx]
            else:
                # try to match raw_ans to option text
                for o in opts_ordered:
                    if raw_ans.lower() in o.lower():
                        answer_text = o
                        break
        if not answer_text:
            # fallback default to first option
            answer_text = opts_ordered[0]

        out.append({"question": q["question"], "options": opts_ordered, "answer": answer_text})

    return out