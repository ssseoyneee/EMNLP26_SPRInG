"""Task-specific helpers: query parsing, output post-processing, small utilities."""
import re
from collections import Counter

name2taskid = {
    "abstract_generation": "LongLaMP_2",
    "review_writing": "LongLaMP_3",
}

TARGET_FIELD = {
    "abstract_generation": "abstract",
    "review_writing": "reviewText",
}


def get_first_k_tokens(text, k):
    return " ".join(text.split()[:k])


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    if count != 0:
        end_list.append(init_list[-count:])
    return end_list


# ---------------------------------------------------------------- query parsing
def extract_abstract_generation(text):
    m = re.search(r'title:\s*["\'](.+?)["\']', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'abstract for\s*["\'](.+?)["\']', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    pos = text.lower().find("title:")
    if pos != -1:
        return text[pos + 6:].strip()
    return text.strip()


_SMART = {"“": '"', "”": '"', "„": '"', "‟": '"', "’": "'", "‚": "'", "´": "'"}


def _norm_quotes(s):
    for a, b in _SMART.items():
        s = s.replace(a, b)
    return s


def extract_review_writing(text):
    """Returns (overall, product, description, summary)."""
    t = _norm_quotes(text)
    flags = re.IGNORECASE | re.DOTALL
    m = re.search(r"for\s+a\s+product\s*'(.+?)'\s+with\s+the\s+following\s+description", t, flags)
    product = m.group(1).strip() if m else "N/A"
    m = re.search(r"with\s+the\s+following\s+description\s*:\s*'(.+?)'\.\s*The\s+review\s+has\s+an\s+overall\s+rating", t, flags)
    description = m.group(1).strip() if m else "N/A"
    m = re.search(r"The\s+review\s+has\s+an\s+overall\s+rating\s+of\s*'(.+?)'", t, flags)
    overall = m.group(1).strip() if m else "N/A"
    m = re.search(r"The\s+review\s+summary\s+is\s*:\s*'(.+?)'", t, flags)
    summary = m.group(1).strip() if m else "N/A"
    return overall, product, description, summary


EXTRACTORS = {
    "abstract_generation": extract_abstract_generation,
    "review_writing": extract_review_writing,
}


def build_query_prompt(task_name, prompt_template, query_input):
    article = EXTRACTORS[task_name](query_input)
    if task_name == "review_writing":
        return prompt_template[task_name]["prompt"].format(*article)
    return prompt_template[task_name]["prompt"].format(article)


# ---------------------------------------------------------- output post-processing
def get_output(output, task_name, model_name):
    model_name = model_name.lower()

    def strip_quotes(text):
        return text.strip().strip('"').strip("'")

    if "mistral" in model_name:
        marker = {"abstract_generation": "Abstract:", "review_writing": "Review:"}[task_name]
        pos = output.rfind(marker)
        return strip_quotes(output[pos + len(marker):].strip() if pos != -1 else "")

    if task_name == "abstract_generation":
        pat = re.compile(r'(?im)^\s*(?:#{0,6}\s*)?(?:\*\*)?\s*abstract\s*(?:\*\*)?\s*[:\-–—]?\s*')
        last = None
        for m in pat.finditer(output):
            last = m
        if last:
            result = output[last.end():].strip()
            nl = result.find("\n")
            first_line = result if nl == -1 else result[:nl]
            if first_line.strip().lower() in {"model", "user", "assistant"}:
                result = ("" if nl == -1 else result[nl + 1:]).lstrip()
        else:
            result = output.strip()

    elif task_name == "review_writing":
        pos = output.rfind("Review:")
        result = output[pos + len("Review:"):].strip() if pos != -1 else output.strip()

    else:
        result = output.strip()

    return strip_quotes(result)


# ------------------------------------- post-processing of interpolation outputs
# Gemma often wraps the answer in a chat preamble ("Okay, here are a few options ...
# **Option 1:** ..."). Interpolation outputs (periods 1-4) are cleaned with
# process_text() after get_output(); this is the same procedure as the original
# process_results.py and is part of the evaluation protocol.
def remove_assistant_prefix(text):
    """Remove leading 'Assistant:' prefixes."""
    return re.compile(r"^(?:assistant:?\s*)+", re.IGNORECASE).sub("", text).strip()


def extract_first_option(text):
    """If the output lists several options ("Option 1", "1."), keep only the first one;
    otherwise drop a leading 'Okay/Here are/Sure,' line."""
    pattern = re.compile(
        r"(?i)(?:(?:\*\*|\#+)?\s*Option\s+1|^\s*1\.)[^\n]*\n+(.*?)(?=\n*\s*(?:(?:\*\*|\#+)?\s*Option\s+2|^\s*2\.|---|To help me)|$)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return re.compile(r"^(Okay|Here are|Sure,)[^\n]*\n+", re.IGNORECASE).sub("", text.strip())


def remove_repetitive_tail(text, min_sentence_len=10, repeat_threshold=3):
    """Cut degenerate repetition: a word/short pattern repeated 5+ times in a row,
    or a sentence that occurs repeat_threshold+ times (keep up to its first occurrence)."""
    if not text:
        return text
    match = re.compile(r"(\b\w+\b\.?\s*)\1{4,}", re.IGNORECASE).search(text)
    if match:
        text = text[:match.start()].strip()

    sentences = re.split(r'(?<=[.?!])\s+', text)
    if len(sentences) <= repeat_threshold:
        return text
    most_common = Counter(s for s in sentences if len(s) > min_sentence_len).most_common(1)
    if most_common and most_common[0][1] >= repeat_threshold:
        first = text.find(most_common[0][0])
        if first != -1:
            return text[:first + len(most_common[0][0])].strip()
    return text


def process_text(raw_text):
    """first option -> repetition cut -> 'Assistant:' prefix removal."""
    return remove_assistant_prefix(remove_repetitive_tail(extract_first_option(raw_text)))
