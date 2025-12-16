import pandas as pd
import torch
import re
import os
from typing import List, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from llama_cpp import Llama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "qwen2.5-7b-instruct-q2_k.gguf")
CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "category_model_v3")
POLARITY_MODEL_PATH = os.path.join(BASE_DIR, "polarity_model")

CATEGORY_BATCH_SIZE = 32
POLARITY_BATCH_SIZE = 64
OPINION_BATCH_SIZE = 32
MAX_NEW_TOKENS_OPINION = 128
MAX_NEW_TOKENS_SPLIT = 256

QWEN_N_GPU_LAYERS = -1
QWEN_N_CTX = 2048
QWEN_N_THREADS = 8
USE_FAST_TOKENIZER = True
DEBUG_MODE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CATEGORY_LABELS = {
    0: "Amenity",
    1: "Branding",
    2: "Experience",
    3: "Facility",
    4: "Loyalty",
    5: "Service",
}

def chat_gguf(model, messages, max_new_tokens=MAX_NEW_TOKENS_SPLIT) -> str:
    output = model.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.1,
        stop=["<|im_end|>"]
    )
    return output['choices'][0]['message']['content']

def split_sentence_with_terms_llm(sentence, model, max_new_tokens=MAX_NEW_TOKENS_SPLIT):
    prompt = (
    "You are an expert linguist working on Aspect-Based Sentiment Analysis (ABSA).\n"
    "Your task is to split the following review sentence into smaller clauses and identify the aspect/term discussed in each clause.\n\n"
    "==================== STRICT RULES ====================\n"
    "1️. DO NOT add, remove, translate, explain, or modify ANY words, symbols, or punctuation in the original sentence.\n"
    "   • Every clause must be a **continuous substring** of the original sentence.\n"
    "   • The output must cover **all parts of the sentence** — no content should be ignored or missing.\n"
    "2️. Only split the sentence where it makes sense semantically — typically around conjunctions ('and', 'but', 'while', 'although', etc.) "
    "or when the opinion changes.\n"
    "   •Do NOT split phrases that grammatically or logically belong to the same subject. "
    "   • If a descriptive phrase does not have a clear term in the sentence, keep it as a separate clause but leave Term blank."
    "3️. Keep the exact original wording and order in each clause. Do NOT reorder, paraphrase, or summarize.\n"
    "4️. Each clause must express a clear **opinion or evaluative meaning**, either explicit (e.g., 'dirty', 'perfect') or implicit "
    "(e.g., 'gave us many tips' implies helpfulness, 'helped us with departure' implies good service).\n"
    "5️. Do NOT separate adverbs (e.g., 'really', 'very', 'so', 'too', 'quite', 'extremely', 'absolutely', "
    "'rather', 'fairly', 'pretty', 'incredibly', 'particularly', 'deeply', 'highly') from the words they modify.\n"
    "6️. Keep negative or limiting words such as 'nothing', 'none', 'nobody', 'no one', 'nowhere', 'never', "
    "'hardly', 'barely', 'scarcely', 'without', 'no', 'not' **inside the same clause** — they must not be removed or separated.\n"
    "7️. Identify the **TERM** being discussed in each clause.\n"
    "   • TERM: the main aspect or entity being described (e.g., 'staff', 'room', 'hotel').\n"
    "   • If no clear term appears, leave it blank.\n"
    "8️. Avoid creating meaningless or redundant clauses.\n"
    "9️. If multiple terms appear in the same clause, separate them with commas.\n"
    "10️. If a clause refers to the same entity as a previous one but does not repeat it explicitly, "
    "**propagate the term from the previous clause**.\n\n"
    "==================== COVERAGE REQUIREMENT ====================\n"
    " Every part of the original sentence must appear in at least one clause.\n"
    " Do NOT skip, shorten, or drop any meaningful phrase, even if it lacks an explicit sentiment word.\n"
    " Clauses that describe actions, experiences, or behaviors with clear positive/negative implications "
    "must be included (e.g., 'gave us many tips', 'helped us with departure').\n\n"
    "==================== OUTPUT FORMAT ====================\n"
    "Clause: <clause text> | Term: <term1,term2,...>\n\n"
    "==================== EXAMPLES ====================\n"
    "Input: The apartment was fully furnished, great facilities, everything was cleaned and well prepared.\n"
    "Output:\n"
    "Clause: The apartment was fully furnished | Term: apartment\n"
    "Clause: great facilities | Term: facilities\n"
    "Clause: everything was cleaned and well prepared | Term: room,facility\n\n"
    "Input: diny was really helpful, he gave us many tips and helped us with departure.\n"
    "Output:\n"
    "Clause: diny was really helpful | Term: staff\n"
    "Clause: he gave us many tips | Term: staff\n"
    "Clause: helped us with departure | Term: staff\n\n"
    "Input: i can definitely recommend it!.\n"
    "Output:\n"
    "Clause: i can definitely recommend it! | Term: \n\n"
    "==================== RESPONSE INSTRUCTION ====================\n"
    "Respond ONLY with the clauses and terms exactly in the format shown above.\n"
    "Do NOT include any explanation, reasoning, or commentary.\n"
    "Do NOT include quotation marks, markdown, or extra text.\n\n"
    f"Now process this sentence WITHOUT changing any words:\n{sentence}"
    )
    messages = [{"role": "user", "content": prompt}]
    response = chat_gguf(model, messages, max_new_tokens=max_new_tokens).strip()

    result = []
    last_term = ""
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "| Term:" in line:
            clause_text, term = line.split("| Term:")
            clause_text = clause_text.replace("Clause:", "").strip()
            term = term.strip()
            if term == "":
                term = last_term
            else:
                last_term = term
        else:
            clause_text = line
            term = last_term
        result.append({"clause": clause_text, "term": term, "sentence_original": sentence})
    return result

def split_and_term_extraction(sentence, model, max_new_tokens=MAX_NEW_TOKENS_SPLIT):
    clauses = split_sentence_with_terms_llm(sentence, model=model, max_new_tokens=max_new_tokens)
    if clauses:
        last_clause = clauses[-1]
        if "term" in last_clause:
            last_clause["term"] = last_clause["term"].replace("<|im_end|>", "").strip()
    for c in clauses:
        terms = [t.strip() for t in c.get("term", "").split(",") if t.strip()]
        terms = [t for t in terms if t.lower() in c["sentence_original"].lower()]
        c["term"] = ",".join(terms)
    return clauses

def make_batch_prompt_en(batch):
    prompt = (
        "You are an expert in Aspect-Based Sentiment Analysis (ABSA).\n\n"
        "Task:\n"
        "For each clause below, extract ALL opinion expressions (adjectives, adverbial phrases, evaluative words/phrases, noun/verb phrases, comparative, superlative, negation, quantity, outcome, result, etc.) that directly describe or evaluate the given term/aspect.\n"
        "Only extract opinion words/phrases that appear EXACTLY in the clause.\n"
        "Keep negations, intensifiers, and quantity words attached. Do NOT paraphrase, translate, or invent.\n"
        "If there is NO opinion, leave the answer BLANK (do NOT write 'None', 'N/A', or any label).\n\n"
        "Strict rules:\n"
        "1. Extract only opinions that clearly describe or evaluate the main term.\n"
        "2. Include adjectives that describe the term (e.g., 'clean', 'friendly', 'comfortable').\n"
        "3. Include adverb + adjective combinations (e.g., 'very helpful', 'extremely rude').\n"
        "4. Include verb phrases that express sentiment (e.g., 'easy to relax', 'hard to find').\n"
        "5. Include outcome/effect expressions that imply sentiment (e.g., 'smooth', 'enjoyable', 'worth the price').\n"
        "6. Include comparative/superlative expressions (e.g., 'better than expected', 'the best').\n"
        "7. Include negated opinions and absence expressions (e.g., 'no hot water', 'not helpful', 'lack of variety').\n"
        "8. Include intensifiers with opinions (e.g., 'too noisy', 'very close', 'at all').\n"
        "9. Include availability/variety/quantity phrases (e.g., 'variety of options', 'plenty of choices').\n"
        "10. If multiple opinion expressions exist, extract ALL of them in their original order.\n"
        "11. If clause describes results/outcomes caused by the term, treat those as opinions about the term.\n"
        "12. For coordinated opinions (e.g., 'clean and comfortable'), SEPARATE them into individual opinions: 'clean, comfortable'.\n"
        "13. REMOVE conjunctions like 'and', 'or', 'but' between opinions. Only keep the core opinion words.\n\n"
        "Special handling:\n"
        "- For fragments without subject/predicate but containing opinion-bearing words, extract those opinion words.\n"
        "- For continuation phrases (e.g., 'at all', 'in our room'), look for modifiers or opinion expressions.\n"
        "- For clauses describing features related to the term (e.g., 'Amazing view from balcony'), extract the descriptive part.\n"
        "- Include 'worth' expressions (e.g., 'totally worth the price', 'worth every penny').\n"
        "- For prepositional phrases with evaluative content (e.g., 'with great service'), extract the evaluative part.\n"
        "- Include exclamatory expressions that convey sentiment (e.g., 'Amazing!', 'Excellent!', 'Terrible!').\n"
        "- Include price-value judgments (e.g., 'overpriced', 'good value').\n\n"
        "Output format: NUMBER. opinion1, opinion2,...\n"
        "Do NOT add any explanation, label, or extra output.\n\n"
        "Data:\n"
    )
    for idx, item in enumerate(batch):
        prompt += f"{idx+1}. Term: '{item.get('term', '')}' | Clause: '{item['clause']}'\n"
    prompt += "\nAnswers:"
    return prompt

def clean_opinion(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'^ID\s*', '', text, flags=re.IGNORECASE)
    text = text.rstrip('.')
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\([^)]+\)', '', text)
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    sentences = text.split(".")
    if len(sentences) > 1:
        text = sentences[0].strip()
    skip_phrases = ["extract", "opinion", "about", "from the clause", "the term", 
                   "describes", "evaluates", "there is no", "output format"]
    for phrase in skip_phrases:
        if phrase in text.lower() and len(text) > 30:
            text = ""
            break
    unwanted = ["opinion:", "answer:", "output:"]
    for word in unwanted:
        text = re.sub(rf'^{word}\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_opinions_only_from_clauses(clauses, model, max_new_tokens=MAX_NEW_TOKENS_OPINION, batch_size=OPINION_BATCH_SIZE):
    if not clauses:
        return clauses
    final_clauses = []
    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i + batch_size]
        prompt_text = make_batch_prompt_en(batch)
        messages = [{"role": "user", "content": prompt_text}]
        response = chat_gguf(model, messages, max_new_tokens=max_new_tokens * batch_size)
        lines = response.strip().split('\n')
        for idx, item in enumerate(batch):
            opinion_extracted = ""
            prefix = f"{idx+1}."
            for line in lines:
                if line.strip().startswith(prefix):
                    content = line.split(prefix, 1)[-1].strip()
                    if not content or content.lower() in ["none", "<none>", "n/a"]:
                        opinion_extracted = ""
                    else:
                        opinion_extracted = content
                    break
            opinion_extracted = clean_opinion(opinion_extracted)
            if opinion_extracted.lower() in ["none", "<none>", "n/a"] or re.match(r'^\d+\.?$', opinion_extracted):
                opinion_extracted = ""
            if opinion_extracted:
                opinions = re.split(r",", opinion_extracted)
                opinions = [o.strip() for o in opinions if o.strip()]
                opinions = [o for o in opinions if len(o.split()) <= 7]
                clause_text = item.get("clause", "")
                sentence_original = item.get("sentence_original", "")
                valid_opinions = []
                for o in opinions:
                    if re.search(rf"\b{re.escape(o)}\b", clause_text, re.IGNORECASE):
                        valid_opinions.append(o)
                    elif re.search(rf"\b{re.escape(o)}\b", sentence_original, re.IGNORECASE):
                        valid_opinions.append(o)
                opinion_extracted = ", ".join(valid_opinions) if valid_opinions else ""
            new_c = {
                "clause": item.get("clause", ""),
                "term": item.get("term", ""),
                "opinion": opinion_extracted,
                "sentence_original": item.get("sentence_original", "")
            }
            final_clauses.append(new_c)
    return final_clauses

def get_predicted_categories(clauses: List[Dict[str, Any]], model, tokenizer, batch_size: int = CATEGORY_BATCH_SIZE) -> List[Dict[str, Any]]:
    MAX_LENGTH = 128
    if not clauses:
        return clauses
    device = next(model.parameters()).device
    model.eval()
    texts = []
    clause_indices = []
    for idx, clause in enumerate(clauses):
        text = str(clause.get("clause", "")).strip()
        if not text:
            clause["category"] = "Unknown"
            clause["category_score"] = 0.0
            continue
        texts.append(text)
        clause_indices.append(idx)
    if not texts:
        return clauses
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_idx = clause_indices[start : start + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                ).to(device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                scores, preds = torch.max(probs, dim=1)
                for idx_local, pred_id, score in zip(batch_idx, preds.tolist(), scores.tolist()):
                    label_name = DEFAULT_CATEGORY_LABELS.get(pred_id, "Unknown")
                    clauses[idx_local]["category"] = label_name
                    clauses[idx_local]["category_score"] = float(score)
    except Exception as exc:
        print(f"[Category Error] {exc}")
        for idx in clause_indices:
            clauses[idx].setdefault("category", "Unknown")
            clauses[idx].setdefault("category_score", 0.0)
    return clauses

def detect_polarity(clauses: List[Dict[str, Any]], polarity_classifier, batch_size=POLARITY_BATCH_SIZE):
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if not clauses:
        return clauses
    texts = [str(c.get("clause", "")).strip() for c in clauses]
    valid_indices = [i for i, t in enumerate(texts) if t]
    valid_texts = [texts[i] for i in valid_indices]
    for i, text in enumerate(texts):
        if not text:
            clauses[i]["polarity"] = "Neutral"
            clauses[i]["polarity_score"] = 0.0
    if not valid_texts:
        return clauses
    try:
        predictions = polarity_classifier(valid_texts, batch_size=batch_size, truncation=True)
        for idx, pred in zip(valid_indices, predictions):
            if isinstance(pred, list):
                pred = max(pred, key=lambda x: x['score'])
            raw_label = pred.get('label', 'Neutral')
            score = pred.get('score', 0.0)
            if raw_label.startswith('LABEL_'):
                label_id = int(raw_label.split('_')[1])
                label = id2label.get(label_id, raw_label)
            else:
                label = raw_label
            clauses[idx]["polarity"] = label.capitalize() if label else "Neutral"
            clauses[idx]["polarity_score"] = round(float(score), 4)
    except Exception as e:
        print(f"[Polarity Error] Batch processing failed: {e}")
        for i in valid_indices:
            clauses[i]["polarity"] = "Neutral"
            clauses[i]["polarity_score"] = 0.0
    return clauses

def load_qwen_model():
    print("Loading Qwen model...")
    model = Llama(
        model_path=QWEN_MODEL_PATH,
        n_gpu_layers=QWEN_N_GPU_LAYERS,
        n_ctx=QWEN_N_CTX,
        n_threads=QWEN_N_THREADS,
        verbose=DEBUG_MODE
    )
    print("Qwen model loaded.")
    return model

def load_category_model():
    print("Loading Category model...")
    tokenizer = AutoTokenizer.from_pretrained(CATEGORY_MODEL_PATH, use_fast=USE_FAST_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(CATEGORY_MODEL_PATH).to(DEVICE).eval()
    print("Category model loaded.")
    return model, tokenizer

def load_polarity_model():
    print("Loading Polarity model...")
    tokenizer = AutoTokenizer.from_pretrained(POLARITY_MODEL_PATH, use_fast=USE_FAST_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(POLARITY_MODEL_PATH)
    polarity_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        device=DEVICE
    )
    print("Polarity model loaded.")
    return polarity_classifier

def run_full_pipeline(sentence: str, qwen_model, cat_model, cat_tokenizer, polarity_classifier) -> pd.DataFrame:
    sentence = (sentence or "").strip()
    if not sentence:
        return pd.DataFrame([])
    clauses_with_details = split_and_term_extraction(sentence, qwen_model, max_new_tokens=MAX_NEW_TOKENS_SPLIT)
    clauses_with_details = extract_opinions_only_from_clauses(clauses_with_details, qwen_model, max_new_tokens=MAX_NEW_TOKENS_OPINION, batch_size=OPINION_BATCH_SIZE)
    clauses_categories = get_predicted_categories(clauses_with_details, cat_model, cat_tokenizer, batch_size=CATEGORY_BATCH_SIZE)
    final_results = detect_polarity(clauses_categories, polarity_classifier, batch_size=POLARITY_BATCH_SIZE)
    df = pd.DataFrame(final_results)
    column_mapping = {
        "clause": "Clause",
        "term": "Term", 
        "opinion": "Opinion",
        "category": "Category",
        "category_score": "Category Score",
        "polarity": "Polarity",
        "polarity_score": "Polarity Score",
        "sentence_original": "Original Sentence"
    }
    df = df.rename(columns=column_mapping)
    columns_order = ["Clause", "Term", "Opinion", "Category", "Category Score", "Polarity", "Polarity Score", "Original Sentence"]
    final_columns = [c for c in columns_order if c in df.columns]
    return df[final_columns] if final_columns else df

def extract_reviews_to_csv(reviews: List[str], output_file: str = "extracted_reviews.csv"):
    print("=" * 60)
    print("ABSA Review Extraction Tool")
    print("=" * 60)
    
    qwen_model = load_qwen_model()
    cat_model, cat_tokenizer = load_category_model()
    polarity_classifier = load_polarity_model()
    
    print(f"\nProcessing {len(reviews)} reviews...")
    print("-" * 60)
    
    all_results = []
    for i, review in enumerate(reviews):
        print(f"[{i+1}/{len(reviews)}] Processing: {review[:50]}...")
        df = run_full_pipeline(review, qwen_model, cat_model, cat_tokenizer, polarity_classifier)
        if not df.empty:
            all_results.append(df)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("-" * 60)
        print(f"Extraction completed! Results saved to: {output_file}")
        print(f"Total rows: {len(final_df)}")
        return final_df
    else:
        print("No results extracted.")
        return pd.DataFrame()

def extract_from_csv_file(input_csv: str, output_csv: str = None, sentence_column: str = "sentence_text"):
    print("=" * 60)
    print("ABSA Review Extraction Tool - CSV Mode")
    print("=" * 60)
    
    input_df = pd.read_csv(input_csv)
    print(f"Loaded {len(input_df)} rows from {input_csv}")
    
    unique_sentences = input_df[sentence_column].dropna().unique().tolist()
    print(f"Found {len(unique_sentences)} unique sentences to process")
    
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(BASE_DIR, f"{base_name}_extracted.csv")
    
    qwen_model = load_qwen_model()
    cat_model, cat_tokenizer = load_category_model()
    polarity_classifier = load_polarity_model()
    
    print(f"\nProcessing {len(unique_sentences)} unique reviews...")
    print("-" * 60)
    
    all_results = []
    for i, review in enumerate(unique_sentences):
        print(f"[{i+1}/{len(unique_sentences)}] Processing: {str(review)[:50]}...")
        df = run_full_pipeline(review, qwen_model, cat_model, cat_tokenizer, polarity_classifier)
        if not df.empty:
            all_results.append(df)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print("-" * 60)
        print(f"Extraction completed! Results saved to: {output_csv}")
        print(f"Total rows: {len(final_df)}")
        return final_df
    else:
        print("No results extracted.")
        return pd.DataFrame()

if __name__ == "__main__":
    input_csv_path = os.path.join(BASE_DIR, "evaluation (6).csv")
    output_csv_path = os.path.join(BASE_DIR, "evaluation_extracted.csv")
    
    df = extract_from_csv_file(input_csv_path, output_csv_path, sentence_column="sentence_text")
    
    if not df.empty:
        print("\nSample output (first 10 rows):")
        print(df.head(10).to_string(index=False))