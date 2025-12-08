import re
from split_clause_lib import chat, chat_gguf
from config import MAX_NEW_TOKENS_OPINION

def extract_opinions_only_from_clauses(clauses, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS_OPINION):
    """
    Extract only OPINION for each clause using LLM.
    If opinion words are not present in the original sentence, exclude them.
    
    Args:
        clauses: Danh sách clause với các thông tin cần thiết.
        model: Mô hình LLM (GGUF).
        tokenizer: Tokenizer (có thể None nếu dùng GGUF).
        max_new_tokens: Số token tối đa cho output.
    
    Returns:
        List[Dict]: Clauses đã được gán thêm khóa 'opinion'.
    """
    if not clauses:
        return clauses
    
    final_clauses = []

    # Xử lý từng clause một
    for c in clauses:
        clause_text = c["clause"]
        term = c.get("term", "")
        sentence_original = c.get("sentence_original", "")
        
        prompt = f"""You are an expert in Aspect-Based Sentiment Analysis (ABSA).

Task: Extract all **opinion expressions** about the aspect/term "{term}" from the following clause.

Strict rules:
1. Only extract opinion words or short opinion phrases that appear **exactly** in the clause.
2. Extract only opinions that clearly describe or evaluate the main term "{term}".
3. Do **NOT** paraphrase, translate, or invent new words.
4. Do **NOT** include explanations, reasoning, or labels.
5. If there is no clear opinion, output an empty string.
6. Output format: comma-separated list — e.g., "very helpful, friendly".
7. If multiple opinion expressions exist, extract **all** of them (including the ones appearing later in the clause) in their original order.
8. If the clause mentions results/outcomes caused by "{term}" (even via pronouns), treat those results as opinions about "{term}" (e.g., "{term} ensured everything was smooth and enjoyable" -> "smooth and enjoyable").
9. Keep negations and intensity words attached to the opinion (e.g., "not helpful at all", "no hot water", "too noisy during the night", "very close").
10. Include availability/variety/quantity phrases that imply evaluation of "{term}" (e.g., "a variety of options", "plenty of choices", "no hot water").

Additional guidance for opinion extraction:
- Include **adjectives** that describe the term (e.g., "clean", "friendly", "comfortable").
- Include **adverb + adjective** combinations (e.g., "very helpful", "extremely rude").
- Include **verb phrases** that express sentiment or evaluation (e.g., "easy to relax", "hard to find").
- Include **outcome/effect expressions** that imply sentiment (e.g., "smooth and enjoyable", "worth the price").
- Include **comparative/superlative** expressions (e.g., "better than expected", "the best").
- If the clause describes a positive/negative experience or result related to "{term}", extract that expression.
- Capture coordinated or sequential opinions joined by "and", "or", or commas; do not drop the final phrase.
- Include **negated** opinions and absence/shortage expressions (e.g., "no hot water", "not helpful", "lack of variety", "without Wi-Fi").
- Include **intensifiers or downtoners** that modify the opinion (e.g., "too noisy", "very close", "at all") and keep them with the opinion phrase.
- Include **nouns/noun phrases** that convey quality/variety/availability when they evaluate the term (e.g., "variety of options", "great service", "poor Wi-Fi connection").

Special handling for incomplete clauses or fragments:
- If the clause is a **fragment without subject/predicate** but contains opinion-bearing words (adjectives, adverbs, evaluative phrases), extract those opinion words.
- If the clause is a **continuation phrase** (e.g., "at all", "in our room", "and not helpful"), look for any **modifier, intensifier, or opinion expression** within it.
- If the clause **describes a feature or attribute** related to "{term}" (e.g., "Amazing view from the balcony"), extract the descriptive/evaluative part as opinion (e.g., "Amazing view").
- If the clause contains **"worth"** expressions (e.g., "totally worth the price", "worth every penny"), extract the full "worth" phrase as opinion.
- If the clause is a **prepositional phrase with evaluative content** (e.g., "with great service", "in perfect condition"), extract the evaluative part.
- If the entire clause **is itself an opinion or sentiment expression** about "{term}", extract the core opinion phrase.
- For clauses like "at all" that are **intensifiers/completers**, output them as they reinforce previous opinions.
- For clauses expressing **location, proximity, or spatial relation** with positive/negative connotation (e.g., "very close to the city center"), extract the evaluative part (e.g., "very close").
- If the clause contains **"from the [term]"** or **"of the [term]"** patterns with a preceding adjective/noun describing quality (e.g., "Amazing view from the balcony"), extract that quality expression.
- Include **exclamatory or emphatic expressions** that convey sentiment (e.g., "Amazing!", "Excellent!", "Terrible!").
- If the clause contains a **price-value judgment** (e.g., "worth the price", "overpriced", "good value"), extract it as opinion about "{term}".
- When in doubt, if any word in the clause carries **positive or negative sentiment**, extract it as opinion.

Clause:
"{clause_text}"

Answer:
"""
        messages = [{"role": "user", "content": prompt}]
        opinion_text = chat_gguf(model, messages, max_new_tokens=max_new_tokens).strip()

        # Làm sạch đầu ra
        opinion_text = (
            opinion_text.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .replace("\n", " ")
            .strip()
        )
        
        # Loại bỏ phần "Answer:" nếu model lặp lại
        if "Answer:" in opinion_text:
            opinion_text = opinion_text.split("Answer:")[-1].strip()
        
        # Loại bỏ các câu giải thích dài
        sentences = opinion_text.split(".")
        if len(sentences) > 1:
            opinion_text = sentences[0].strip()
        
        # Loại bỏ các từ khóa của instructions
        skip_phrases = ["extract", "opinion", "about", "from the clause", "the term", 
                       "describes", "evaluates", "there is no", "output format"]
        for phrase in skip_phrases:
            if phrase in opinion_text.lower() and len(opinion_text) > 30:
                opinion_text = ""
                break

        # Chuẩn hóa danh sách opinions
        opinions = re.split(r",", opinion_text)
        opinions = [o.strip() for o in opinions if o.strip()]

        # Lọc bỏ những "opinion" quá dài (có thể là explanation)
        opinions = [o for o in opinions if len(o.split()) <= 7]
        
        # Chỉ giữ opinions xuất hiện trong clause hoặc sentence gốc
        valid_opinions = []
        for o in opinions:
            if re.search(rf"\b{re.escape(o)}\b", clause_text, re.IGNORECASE):
                valid_opinions.append(o)
            elif re.search(rf"\b{re.escape(o)}\b", sentence_original, re.IGNORECASE):
                valid_opinions.append(o)

        new_c = c.copy()
        new_c["opinion"] = ", ".join(valid_opinions) if valid_opinions else ""
        final_clauses.append(new_c)

    return final_clauses