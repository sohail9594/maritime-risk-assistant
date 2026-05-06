"""Free local RAG pipeline for the maritime chatbot.

This version does not use OpenAI or any paid API.

Before running this script:
1. Install the project requirements:
   pip install -r requirements.txt

2. Run the script:
   python3 utils/rag_pipeline.py

Note:
The first run may download the free Sentence Transformers model. After that,
the model is cached locally on your machine.
"""

import os
import re
from pathlib import Path
from typing import Optional, Union

# This project only uses Sentence Transformers with PyTorch. Some Python
# environments also have TensorFlow/Keras installed, which can make Transformers
# import optional TensorFlow modules and fail with Keras 3. Disable TensorFlow
# before importing sentence_transformers so the local RAG stack stays simple.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

try:
    from utils.routing import is_route_query, suggest_route
except ImportError:
    from routing import is_route_query, suggest_route

try:
    from utils.logistics import estimate_trucking_time_from_query, is_trucking_time_query
except ImportError:
    from logistics import estimate_trucking_time_from_query, is_trucking_time_query

try:
    from utils.decision_logic import (
        answer_fragile_electronics_decision,
        answer_iot_satellite_jamming_decision,
        answer_medical_diversion_decision,
        is_fragile_electronics_transport_query,
        is_iot_satellite_jamming_query,
        is_medical_diversion_decision_query,
    )
except ImportError:
    from decision_logic import (
        answer_fragile_electronics_decision,
        answer_iot_satellite_jamming_decision,
        answer_medical_diversion_decision,
        is_fragile_electronics_transport_query,
        is_iot_satellite_jamming_query,
        is_medical_diversion_decision_query,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LEGAL_SOURCE_FILENAME = "uae_federal_maritime_law.txt"
LEGAL_DISCLAIMER = "Note: This is an educational prototype and not legal advice."
INSURANCE_DISCLAIMER = (
    "Note: This is an educational prototype and not insurance advice."
)
GENERAL_CATEGORY = "general"
UAE_LAW_CATEGORY = "uae_law"
INTERNATIONAL_LAW_CATEGORY = "international_law"
INSURANCE_CATEGORY = "insurance"
LEGAL_CATEGORIES = {UAE_LAW_CATEGORY, INTERNATIONAL_LAW_CATEGORY}
REFERENCE_CATEGORIES = LEGAL_CATEGORIES | {INSURANCE_CATEGORY}

SYSTEM_PROMPT = (
    "You are a Maritime Risk Analyst. Answer only using the provided context. "
    "If unsure, say you don't have enough data."
)

STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "can",
    "for",
    "from",
    "have",
    "how",
    "into",
    "near",
    "should",
    "that",
    "the",
    "their",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
}

RISK_KEYWORDS = {
    "advisory",
    "congestion",
    "delay",
    "delays",
    "disruption",
    "insurance",
    "premium",
    "premiums",
    "risk",
    "risks",
    "security",
    "surcharge",
    "war-risk",
}

RECOMMENDATION_KEYWORDS = {
    "advised",
    "alternative",
    "avoid",
    "compare",
    "confirm",
    "consider",
    "monitor",
    "recommended",
    "review",
    "route",
    "routes",
    "routing",
    "should",
    "suitable",
    "update",
    "verify",
}

ACTION_KEYWORDS = {
    "avoid",
    "confirm",
    "consider",
    "monitor",
    "review",
    "use",
    "verify",
}

LEGAL_QUERY_TERMS = {
    "cargo ownership",
    "carrier responsibility",
    "compliance",
    "customs",
    "detention",
    "documentation",
    "law",
    "legal",
    "liability",
    "maritime law",
    "port authority",
    "regulatory risk",
    "sanctions",
    "seizure",
}

UAE_QUERY_TERMS = {
    "abu dhabi",
    "dubai",
    "jebel ali",
    "uae",
    "united arab emirates",
}

INTERNATIONAL_QUERY_TERMS = {
    "coastal state",
    "convention",
    "foreign",
    "global",
    "high seas",
    "imo",
    "innocent passage",
    "international",
    "marpol",
    "solas",
    "territorial waters",
    "treaty",
    "unclos",
    "worldwide",
}

TREATY_LEGAL_SIGNAL_TERMS = {
    "coastal state",
    "convention",
    "high seas",
    "imo",
    "innocent passage",
    "marpol",
    "solas",
    "territorial waters",
    "treaty",
    "unclos",
}

INSURANCE_QUERY_TERMS = {
    "cargo damage",
    "claim",
    "claims",
    "compensation",
    "coverage",
    "delay compensation",
    "exclusion",
    "exclusions",
    "hull value",
    "insurance",
    "insured",
    "insurer",
    "liability",
    "premium",
    "premiums",
    "policy",
    "seizure",
    "war risk",
    "war-risk",
}

INSURANCE_STRONG_QUERY_TERMS = {
    "cargo damage",
    "claim",
    "claims",
    "compensation",
    "coverage",
    "delay compensation",
    "exclusion",
    "exclusions",
    "hull value",
    "insurance",
    "insured",
    "insurer",
    "p&i",
    "p and i",
    "policy",
    "premium",
    "premiums",
    "protection and indemnity",
    "war risk",
    "war-risk",
}

LEGAL_LOW_VALUE_TERMS = {
    "about",
    "apply",
    "applies",
    "available",
    "compliance",
    "decree",
    "federal",
    "legal",
    "law",
    "maritime",
    "point",
    "provide",
    "regarding",
    "regulatory",
    "risk",
    "under",
    "uae",
}

INSURANCE_LOW_VALUE_TERMS = {
    "about",
    "apply",
    "available",
    "insurance",
    "insured",
    "insurer",
    "maritime",
    "point",
    "provide",
    "under",
}

LEGAL_TERM_VARIANTS = {
    "authority": {"authorities", "authority"},
    "carrier": {"carrier", "carriers"},
    "cargo": {"cargo", "cargoes"},
    "customs": {"customs"},
    "detention": {"detained", "detention"},
    "documentation": {"document", "documentation", "documents"},
    "insurance": {"insured", "insurer", "insurance"},
    "liability": {"liability", "liable"},
    "ownership": {"owner", "ownership"},
    "seizure": {"seized", "seizure"},
}

INSURANCE_TERM_VARIANTS = {
    "claim": {"claim", "claims"},
    "compensation": {"compensation", "compensate"},
    "coverage": {"coverage", "cover", "covered"},
    "damage": {"damage", "damaged", "damages"},
    "exclusion": {"excluded", "exclusion", "exclusions"},
    "hull": {"hull"},
    "liability": {"liability", "liable"},
    "premium": {"premium", "premiums"},
    "seizure": {"seized", "seizure"},
}

LEGAL_BOILERPLATE_PHRASES = {
    "federal decree by law no 43 of 2023 concerning the maritime law",
    "official gazette",
    "issued by us mohammed bin zayed",
}

INCOMPLETE_SENTENCE_STARTS = {
    "and",
    "but",
    "by",
    "for",
    "from",
    "in",
    "of",
    "or",
    "than",
    "to",
    "with",
}

INCOMPLETE_SENTENCE_ENDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "in",
    "of",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def get_document_category(file_name: str) -> str:
    """Assign a simple category from the filename."""
    clean_file_name = file_name.lower()

    if clean_file_name == LEGAL_SOURCE_FILENAME:
        return UAE_LAW_CATEGORY

    if clean_file_name.startswith("uae_") and "law" in clean_file_name:
        return UAE_LAW_CATEGORY

    if clean_file_name.startswith("international_maritime_law_"):
        return INTERNATIONAL_LAW_CATEGORY

    if clean_file_name.startswith("maritime_insurance_"):
        return INSURANCE_CATEGORY

    return GENERAL_CATEGORY


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper for a local Sentence Transformers model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document chunks into local embeddings."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Convert a user question into a local embedding."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


def load_documents() -> list[Document]:
    """Load all .txt files from the data folder as LangChain documents."""
    documents = []

    if not DATA_FOLDER.exists():
        print("Data folder not found. Please create a folder named 'data'.")
        return documents

    txt_files = sorted(DATA_FOLDER.glob("*.txt"))

    if not txt_files:
        print("No .txt files found in the data folder.")
        return documents

    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"Skipped missing file: {file_path.name}")
            continue

        if not text:
            print(f"Skipped empty file: {file_path.name}")
            continue

        category = get_document_category(file_path.name)
        metadata = {
            "source": file_path.name,
            "category": category,
        }

        if category in LEGAL_CATEGORIES:
            metadata["document_type"] = "legal_reference"
            metadata["priority"] = "high"
        elif category == INSURANCE_CATEGORY:
            metadata["document_type"] = "insurance_reference"
            metadata["priority"] = "high"

        document = Document(
            page_content=text,
            metadata=metadata,
        )
        documents.append(document)

    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks


def create_vector_database(chunks: list[Document]) -> Chroma:
    """Create a ChromaDB vector database using local embeddings."""
    embeddings = LocalSentenceTransformerEmbeddings()

    vector_database = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="maritime_documents",
    )

    print("Created ChromaDB vector database with local embeddings.")
    return vector_database


def build_rag_pipeline():
    """Load documents, split them, embed them locally, and return a retriever."""
    documents = load_documents()
    if not documents:
        return None

    chunks = split_documents(documents)
    vector_database = create_vector_database(chunks)

    retriever = vector_database.as_retriever(search_kwargs={"k": 3})
    return retriever


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation rules."""
    compact_text = " ".join(text.split())
    details_match = re.search(r"\bDETAILS:\s*(.+)", compact_text, flags=re.I)

    if details_match:
        compact_text = details_match.group(1).strip()

    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", compact_text)]


def normalize_text(text: str) -> str:
    """Normalize text so duplicate sentences are easier to detect."""
    lowercase_text = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowercase_text).strip()


def get_question_terms(question: str) -> set[str]:
    """Get important words from the question for simple relevance scoring."""
    words = re.findall(r"[a-zA-Z][a-zA-Z-]+", question.lower())
    return {word for word in words if len(word) > 3 and word not in STOP_WORDS}


def is_legal_query(question: str) -> bool:
    """Detect legal, compliance, liability, or regulatory questions."""
    clean_question = normalize_text(question)

    return any(normalize_text(term) in clean_question for term in LEGAL_QUERY_TERMS)


def is_uae_specific_query(question: str) -> bool:
    """Detect UAE-specific legal or regulatory questions."""
    clean_question = normalize_text(question)
    return any(normalize_text(term) in clean_question for term in UAE_QUERY_TERMS)


def is_international_legal_query(question: str) -> bool:
    """Detect international or treaty-based maritime law questions."""
    clean_question = normalize_text(question)
    return any(
        normalize_text(term) in clean_question
        for term in INTERNATIONAL_QUERY_TERMS
    )


def is_insurance_query(question: str) -> bool:
    """Detect insurance, claims, premium, coverage, or cargo damage questions."""
    clean_question = normalize_text(question)
    return any(
        normalize_text(term) in clean_question
        for term in INSURANCE_STRONG_QUERY_TERMS
    )


def get_priority_categories(question: str) -> list[str]:
    """Choose specialized document categories for legal/insurance questions."""
    categories = []
    clean_question = normalize_text(question)
    treaty_legal_signal = any(
        normalize_text(term) in clean_question
        for term in TREATY_LEGAL_SIGNAL_TERMS
    )
    legal_query = is_legal_query(question) or treaty_legal_signal
    insurance_query = is_insurance_query(question)

    if legal_query:
        if is_uae_specific_query(question):
            categories.append(UAE_LAW_CATEGORY)

        if is_international_legal_query(question):
            categories.append(INTERNATIONAL_LAW_CATEGORY)

        if not categories:
            categories.extend([UAE_LAW_CATEGORY, INTERNATIONAL_LAW_CATEGORY])

    if insurance_query:
        categories.append(INSURANCE_CATEGORY)

    unique_categories = []
    for category in categories:
        if category not in unique_categories:
            unique_categories.append(category)

    return unique_categories


def is_reference_priority_query(question: str) -> bool:
    """Return True for legal or insurance questions needing priority sources."""
    return bool(get_priority_categories(question))


def is_legal_document(document: Document) -> bool:
    """Check whether a retrieved document chunk comes from the law file."""
    return document.metadata.get("category") in LEGAL_CATEGORIES


def is_reference_document(document: Document, categories: list[str]) -> bool:
    """Check whether a document belongs to one of the priority categories."""
    return document.metadata.get("category") in categories


def get_legal_focus_terms(question: str) -> set[str]:
    """Keep only the legal words that describe the user's exact issue."""
    focus_terms = {
        term
        for term in get_question_terms(question)
        if term not in LEGAL_LOW_VALUE_TERMS
    }
    clean_question = normalize_text(question)

    for term in LEGAL_QUERY_TERMS:
        normalized_term = normalize_text(term)

        if normalized_term in clean_question:
            focus_terms.update(
                word
                for word in normalized_term.split()
                if word not in LEGAL_LOW_VALUE_TERMS
            )

    expanded_terms = set(focus_terms)
    for term in focus_terms:
        expanded_terms.update(LEGAL_TERM_VARIANTS.get(term, {term}))

    return expanded_terms


def get_insurance_focus_terms(question: str) -> set[str]:
    """Keep only insurance words that describe the user's exact issue."""
    focus_terms = {
        term
        for term in get_question_terms(question)
        if term not in INSURANCE_LOW_VALUE_TERMS
    }
    clean_question = normalize_text(question)

    for term in INSURANCE_QUERY_TERMS:
        normalized_term = normalize_text(term)

        if normalized_term in clean_question:
            focus_terms.update(
                word
                for word in normalized_term.split()
                if word not in INSURANCE_LOW_VALUE_TERMS
            )

    expanded_terms = set(focus_terms)
    for term in focus_terms:
        expanded_terms.update(INSURANCE_TERM_VARIANTS.get(term, {term}))

    return expanded_terms


def get_reference_focus_terms(question: str, categories: list[str]) -> set[str]:
    """Get relevance terms for the selected legal and insurance categories."""
    focus_terms = set()

    if any(category in LEGAL_CATEGORIES for category in categories):
        focus_terms.update(get_legal_focus_terms(question))

    if INSURANCE_CATEGORY in categories:
        focus_terms.update(get_insurance_focus_terms(question))

    return focus_terms


def is_boilerplate_sentence(sentence: str) -> bool:
    """Avoid using document titles or page headers as legal evidence."""
    if re.search(r"\b(DATE|SOURCE|SUBJECT|DOCUMENT_TYPE):", sentence, flags=re.I):
        return True

    normalized_sentence = normalize_text(sentence)
    return any(phrase in normalized_sentence for phrase in LEGAL_BOILERPLATE_PHRASES)


def clean_sentence(sentence: str) -> str:
    """Clean extracted text into one complete, readable sentence."""
    cleaned = " ".join(sentence.split())
    simple_pdf_fixes = {
        "Decree -Law": "Decree-Law",
        "vis -à-vis": "vis-à-vis",
        "non -": "non-",
        "wit h": "with",
        "w ith": "with",
        "insuranc e": "insurance",
        "compe tent": "competent",
        "repre sentative": "representative",
        "relat ed": "related",
        "C lause": "Clause",
    }

    for broken_text, fixed_text in simple_pdf_fixes.items():
        cleaned = cleaned.replace(broken_text, fixed_text)

    cleaned = re.sub(r"^(details|subject|source|alert|advisory):\s*", "", cleaned, flags=re.I)
    cleaned = cleaned.strip(" -;:,")

    if not cleaned:
        return ""

    if cleaned[-1] not in ".!?":
        cleaned += "."

    return cleaned[0].upper() + cleaned[1:]


def looks_incomplete_sentence(sentence: str) -> bool:
    """Detect text fragments created when retrieval chunks start mid-sentence."""
    stripped = sentence.strip()

    if not stripped:
        return True

    first_word_match = re.match(r"[A-Za-z-]+", stripped)
    first_word = first_word_match.group(0).lower() if first_word_match else ""

    if first_word in INCOMPLETE_SENTENCE_STARTS:
        return True

    if stripped[0].islower():
        return True

    last_word_match = re.search(r"[A-Za-z-]+[.!?]?$", stripped)
    last_word = last_word_match.group(0).strip(".!?").lower() if last_word_match else ""

    if last_word in INCOMPLETE_SENTENCE_ENDS:
        return True

    return False


def shorten_complete_sentence(sentence: str, max_words: int = 60) -> str:
    """Keep text concise without cutting sentences mid-thought."""
    cleaned = clean_sentence(sentence)

    if len(cleaned.split()) <= max_words:
        return cleaned

    clauses = re.split(r",\s+|;\s+|\s+but\s+", cleaned)
    short_clauses = []
    word_count = 0

    for clause in clauses:
        clause_words = clause.strip().split()

        if not clause_words:
            continue

        if word_count + len(clause_words) > max_words and short_clauses:
            break

        short_clauses.append(clause.strip().rstrip(".!?"))
        word_count += len(clause_words)

    if short_clauses:
        return clean_sentence("; ".join(short_clauses))

    return cleaned


def is_duplicate_or_overlap(text: str, excluded_texts: set[str]) -> bool:
    """Check for exact or near-duplicate normalized text."""
    normalized_text = normalize_text(text)

    for excluded_text in excluded_texts:
        if not excluded_text:
            continue

        if normalized_text == excluded_text:
            return True

        if normalized_text.startswith(excluded_text) or excluded_text.startswith(normalized_text):
            return True

    return False


def collect_unique_sentences(
    retrieved_documents: list[Document],
) -> list[dict[str, Union[str, int]]]:
    """Collect non-duplicate sentences from retrieved chunks."""
    unique_sentences = []
    seen_sentences = set()

    for document_index, document in enumerate(retrieved_documents):
        source = document.metadata.get("source", "Unknown source")

        for sentence_index, sentence in enumerate(split_into_sentences(document.page_content)):
            if looks_incomplete_sentence(sentence):
                continue

            cleaned_sentence = clean_sentence(sentence)
            normalized_sentence = normalize_text(cleaned_sentence)

            if is_boilerplate_sentence(cleaned_sentence):
                continue

            if len(normalized_sentence) < 30 or normalized_sentence in seen_sentences:
                continue

            seen_sentences.add(normalized_sentence)
            unique_sentences.append(
                {
                    "text": cleaned_sentence,
                    "source": source,
                    "document_index": document_index,
                    "sentence_index": sentence_index,
                }
            )

    return unique_sentences


def score_sentence(
    sentence_info: dict[str, Union[str, int]],
    question_terms: set[str],
    preferred_keywords: set[str],
) -> float:
    """Score a sentence using question words and useful maritime keywords."""
    sentence_text = str(sentence_info["text"]).lower()
    sentence_words = set(re.findall(r"[a-zA-Z][a-zA-Z-]+", sentence_text))

    question_score = len(sentence_words & question_terms) * 3
    preferred_score = len(sentence_words & preferred_keywords) * 2
    risk_score = len(sentence_words & RISK_KEYWORDS)

    # Earlier retrieved documents are usually more relevant, so give them a
    # small boost without making this logic complicated.
    retrieval_boost = 1 / (int(sentence_info["document_index"]) + 1)

    return question_score + preferred_score + risk_score + retrieval_boost


def choose_best_sentence(
    sentences: list[dict[str, Union[str, int]]],
    question_terms: set[str],
    preferred_keywords: set[str],
    excluded_sentences: Optional[set[str]] = None,
) -> str:
    """Pick one concise sentence from retrieved context."""
    if not sentences:
        return "I don't have enough data."

    excluded_sentences = excluded_sentences or set()
    ranked_sentences = sorted(
        sentences,
        key=lambda sentence_info: score_sentence(
            sentence_info,
            question_terms,
            preferred_keywords,
        ),
        reverse=True,
    )

    for sentence_info in ranked_sentences:
        sentence_text = str(sentence_info["text"])

        if is_duplicate_or_overlap(sentence_text, excluded_sentences):
            continue

        return shorten_complete_sentence(sentence_text)

    return "I don't have enough data."


def make_recommendation(sentence: str) -> str:
    """Turn selected context into a clear local recommendation."""
    cleaned = shorten_complete_sentence(sentence)
    normalized = normalize_text(cleaned)

    if cleaned == "I don't have enough data.":
        return cleaned

    if normalized.startswith(("avoid ", "use ", "monitor ", "verify ", "review ", "confirm ")):
        return cleaned

    if any(keyword in normalized.split() for keyword in ACTION_KEYWORDS):
        return cleaned

    return f"Use this information to choose the lower-risk option: {cleaned}"


def get_source_names(retrieved_documents: list[Document]) -> list[str]:
    """Return up to three unique source filenames."""
    source_names = []

    for document in retrieved_documents:
        source = document.metadata.get("source", "Unknown source")

        if source not in source_names:
            source_names.append(source)

        if len(source_names) == 3:
            break

    return source_names


def format_source_lines(source_names: list[str]) -> str:
    """Format source filenames with one unique source per line."""
    clean_source_names = []

    for source in source_names:
        if source and source not in clean_source_names:
            clean_source_names.append(source)

        if len(clean_source_names) == 3:
            break

    if not clean_source_names:
        clean_source_names = ["No matching source found"]

    return "\n".join(f"- {source}" for source in clean_source_names)


def score_reference_chunk(chunk: Document, focus_terms: set[str]) -> int:
    """Score a priority reference chunk using simple question-word overlap."""
    chunk_text = normalize_text(chunk.page_content)
    chunk_words = set(chunk_text.split())

    if not focus_terms:
        return 0

    if "carrier" in focus_terms and not ({"carrier", "carriers"} & chunk_words):
        return 0

    matching_terms = chunk_words & focus_terms
    score = len(matching_terms) * 4

    for term in focus_terms:
        score += min(chunk_text.count(term), 5)

    return score


def filter_reference_sentences_by_focus(
    sentences: list[dict[str, Union[str, int]]],
    focus_terms: set[str],
) -> list[dict[str, Union[str, int]]]:
    """Keep evidence tied to the user's most important legal/insurance terms."""
    filtered_sentences = [
        sentence_info
        for sentence_info in sentences
        if set(normalize_text(str(sentence_info["text"])).split()) & focus_terms
    ]

    priority_term_groups = [
        {"carrier", "carriers"},
        {"liability", "liable"},
        {"insurance", "insured", "insurer"},
        {"claim", "claims"},
        {"premium", "premiums"},
        {"coverage", "cover", "covered"},
        {"exclusion", "exclusions", "excluded"},
        {"damage", "damages", "damaged"},
        {"compensation", "compensate"},
        {"war", "war-risk"},
        {"hull"},
        {"customs"},
        {"detention", "detained"},
        {"seizure", "seized"},
    ]

    for term_group in priority_term_groups:
        if not (focus_terms & term_group):
            continue

        stronger_matches = [
            sentence_info
            for sentence_info in filtered_sentences
            if set(normalize_text(str(sentence_info["text"])).split()) & term_group
        ]

        if stronger_matches:
            filtered_sentences = stronger_matches

    return filtered_sentences


def select_relevant_reference_chunks(
    chunks: list[Document],
    focus_terms: set[str],
    limit: int = 2,
) -> list[Document]:
    """Return only priority chunks that match the user's specific point."""
    if not focus_terms:
        return []

    scored_chunks = [
        (score_reference_chunk(chunk, focus_terms), chunk)
        for chunk in chunks
    ]
    relevant_chunks = [
        chunk
        for score, chunk in sorted(scored_chunks, key=lambda item: item[0], reverse=True)
        if score >= 4
    ]

    return relevant_chunks[:limit]


def load_reference_documents_by_category(category: str) -> list[Document]:
    """Load all local documents for one priority category."""
    if not DATA_FOLDER.exists():
        return []

    documents = []

    for file_path in sorted(DATA_FOLDER.glob("*.txt")):
        if get_document_category(file_path.name) != category:
            continue

        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue

        if not text:
            continue

        metadata = {
            "source": file_path.name,
            "category": category,
            "priority": "high",
        }

        if category in LEGAL_CATEGORIES:
            metadata["document_type"] = "legal_reference"
        elif category == INSURANCE_CATEGORY:
            metadata["document_type"] = "insurance_reference"

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def split_reference_documents(documents: list[Document]) -> list[Document]:
    """Split priority reference documents for focused fallback search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
    )
    return text_splitter.split_documents(documents)


def find_priority_reference_chunks(
    question: str,
    retrieved_documents: list[Document],
    categories: list[str],
) -> list[Document]:
    """Find relevant legal/insurance chunks without changing vector retrieval."""
    focus_terms = get_reference_focus_terms(question, categories)

    if not focus_terms:
        return []

    selected_chunks = []

    for category in categories:
        retrieved_chunks = [
            document
            for document in retrieved_documents
            if document.metadata.get("category") == category
        ]
        category_matches = select_relevant_reference_chunks(
            retrieved_chunks,
            focus_terms,
        )

        if not category_matches:
            fallback_documents = load_reference_documents_by_category(category)
            fallback_chunks = split_reference_documents(fallback_documents)
            category_matches = select_relevant_reference_chunks(
                fallback_chunks,
                focus_terms,
            )

        selected_chunks.extend(category_matches)

    return selected_chunks[:6]


def get_reference_disclaimers(categories: list[str]) -> list[str]:
    """Return the needed disclaimers for selected categories."""
    disclaimers = []

    if any(category in LEGAL_CATEGORIES for category in categories):
        disclaimers.append(LEGAL_DISCLAIMER)

    if INSURANCE_CATEGORY in categories:
        disclaimers.append(INSURANCE_DISCLAIMER)

    return disclaimers


def make_not_enough_reference_answer(categories: list[str]) -> str:
    """Return the required cautious fallback for legal/insurance questions."""
    answer_lines = [
        "Risk Analysis:",
        "- The available legal/insurance documents do not provide enough information to answer this point.",
        "Recommendation:",
        "- Do not rely on this prototype for the decision; seek qualified review.",
        "Sources:",
        format_source_lines([]),
    ]

    answer_lines.extend(get_reference_disclaimers(categories))
    return "\n".join(answer_lines)


def make_reference_recommendation(categories: list[str]) -> str:
    """Create a cautious action line for legal/insurance answers."""
    has_legal = any(category in LEGAL_CATEGORIES for category in categories)
    has_insurance = INSURANCE_CATEGORY in categories

    if has_legal and has_insurance:
        return (
            "Use the cited legal and insurance text as the starting point and "
            "get qualified review before acting."
        )

    if has_insurance:
        return (
            "Use the cited insurance text as the starting point and confirm "
            "coverage with a qualified broker or insurer before acting."
        )

    return (
        "Use the cited legal text as the starting point and get qualified legal "
        "review before acting."
    )


def generate_reference_answer(
    question: str,
    retrieved_documents: list[Document],
) -> str:
    """Generate a cautious answer from priority legal/insurance sources."""
    categories = get_priority_categories(question)

    if not categories:
        return generate_local_answer(question, retrieved_documents)

    reference_chunks = find_priority_reference_chunks(
        question,
        retrieved_documents,
        categories,
    )

    if not reference_chunks:
        return make_not_enough_reference_answer(categories)

    focus_terms = get_reference_focus_terms(question, categories)
    unique_sentences = filter_reference_sentences_by_focus(
        collect_unique_sentences(reference_chunks),
        focus_terms,
    )

    if not unique_sentences:
        return make_not_enough_reference_answer(categories)

    preferred_keywords = set()

    if any(category in LEGAL_CATEGORIES for category in categories):
        preferred_keywords.update(LEGAL_QUERY_TERMS)

    if INSURANCE_CATEGORY in categories:
        preferred_keywords.update(INSURANCE_QUERY_TERMS)

    reference_summary = choose_best_sentence(
        unique_sentences,
        focus_terms,
        preferred_keywords,
    )
    recommendation = make_reference_recommendation(categories)
    source_names = get_source_names(reference_chunks)

    answer_lines = [
        "Risk Analysis:",
        f"- {reference_summary}",
        "Recommendation:",
        f"- {recommendation}",
        "Sources:",
        format_source_lines(source_names),
    ]

    answer_lines.extend(get_reference_disclaimers(categories))
    return "\n".join(answer_lines)


def generate_legal_answer(
    question: str,
    retrieved_documents: list[Document],
) -> str:
    """Compatibility wrapper for older code paths."""
    return generate_reference_answer(question, retrieved_documents)


def generate_local_answer(question: str, retrieved_documents: list[Document]) -> str:
    """Create a concise, structured answer from retrieved context only.

    This replaces the paid LLM call with simple local string processing:
    sentence splitting, duplicate removal, relevance scoring, and formatting.
    """
    if not retrieved_documents:
        return "I don't have enough data."

    question_terms = get_question_terms(question)
    unique_sentences = collect_unique_sentences(retrieved_documents)

    if not unique_sentences:
        return "I don't have enough data."

    risk_summary = choose_best_sentence(
        unique_sentences,
        question_terms,
        RISK_KEYWORDS,
    )
    recommendation = choose_best_sentence(
        unique_sentences,
        question_terms,
        RECOMMENDATION_KEYWORDS,
        excluded_sentences={normalize_text(risk_summary)},
    )
    recommendation = make_recommendation(recommendation)

    if normalize_text(risk_summary) == normalize_text(recommendation):
        recommendation = "Review the retrieved advisory before proceeding and choose the lower-risk option."

    source_names = get_source_names(retrieved_documents)

    answer_lines = [
        "Risk Analysis:",
        f"- {risk_summary}",
        "Recommendation:",
        f"- {recommendation}",
        "Sources:",
    ]

    answer_lines.append(format_source_lines(source_names))

    return "\n".join(answer_lines)


def answer_question(question: str, retriever=None) -> str:
    """Retrieve relevant chunks and generate a local answer."""
    if not question.strip():
        return "Please enter a question."

    # Domain-specific decision logic runs before RAG so specialized cargo
    # questions do not get generic or irrelevant recommendations.
    if is_fragile_electronics_transport_query(question):
        return answer_fragile_electronics_decision(question)

    if is_iot_satellite_jamming_query(question):
        return answer_iot_satellite_jamming_decision(question)

    if is_medical_diversion_decision_query(question):
        return answer_medical_diversion_decision(question)

    if retriever is None:
        if is_reference_priority_query(question):
            retriever = build_rag_pipeline()
        elif not is_route_query(question) and not is_trucking_time_query(question):
            retriever = build_rag_pipeline()

    if is_reference_priority_query(question):
        if retriever is None:
            return generate_reference_answer(question, [])

        retrieved_documents = retriever.invoke(question)
        return generate_reference_answer(question, retrieved_documents)

    if is_route_query(question):
        return suggest_route(question)

    if is_trucking_time_query(question):
        return estimate_trucking_time_from_query(question)

    if retriever is None:
        return "I don't have enough data."

    retrieved_documents = retriever.invoke(question)
    return generate_local_answer(question, retrieved_documents)


def run_cli() -> None:
    """Run a simple terminal chat loop."""
    retriever = None

    print("Local maritime chatbot is ready. Type 'exit' to stop.")
    print("No API key is required.")

    while True:
        question = input("\nAsk a maritime question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        needs_retriever = (
            is_reference_priority_query(question)
            or (
                not is_fragile_electronics_transport_query(question)
                and not is_iot_satellite_jamming_query(question)
                and not is_medical_diversion_decision_query(question)
                and not is_route_query(question)
                and not is_trucking_time_query(question)
            )
        )

        if needs_retriever and retriever is None:
            print("Building free local maritime RAG pipeline...")
            retriever = build_rag_pipeline()

            if retriever is None:
                print("No documents are available for retrieval.")
                continue

        answer = answer_question(question, retriever)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    run_cli()
