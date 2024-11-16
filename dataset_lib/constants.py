from enum import Enum


class SumDomains(Enum):
    SCIENTIFIC = "scientific"
    MEDICAL = "medical"
    LEGAL = "legal"
    NEWS = "news"
    UNGROUPED = "ungrouped"


DEFAULT_SYSTEM_PROMPT = """
    Summarize the given article by addressing the following key points:

    1. Main Topic: What is the primary subject or theme of the article?

    2. Context: What background information is provided to frame the discussion?

    3. Key Arguments: What are the main arguments or points made by the author?

    4. Evidence and Examples: What supporting evidence or examples are presented?

    5. Conclusions: What conclusions does the author draw from the discussion?

    6. Implications: What are the potential implications or significance of the article's content for its audience?
""".strip()

SAMPLE_PROMPT = """ 
    Summarize the given article by including the following key points:

    1. Objective: What is the main research question or objective of the study?

    2. Background: What is the context or rationale for the study?

    3. Methods: What study design, population, and methodologies were used?

    4. Key Findings: What are the most significant results or discoveries from the study?

    5. Conclusions: What conclusions do the authors draw from their findings?

    6. Clinical Relevance: How might the study’s findings impact medical practice or patient care?
    """


datasets_info_dict = {
    SumDomains.SCIENTIFIC: {
        "arxiv": {
            "dataset_id": "ccdv/arxiv-summarization",
            "local_path": "domains/scientific/arxiv",
            "version": None,
            "columns_to_remove": ["article", "abstract"],
            "source": "hugging_face"
        },
        "ssn": {

        },
        "elsevier": {
            "dataset_id": "orieg/elsevier-oa-cc-by",
            "local_path": "domains/scientific/elsevier",
            "version": None,
            "columns_to_remove": ["title", "abstract", "subjareas", "keywords", "asjc", "body_text", "author_highlights"],
            "source": "hugging_face"
        },
        "scitldr": {
            "dataset_id": "allenai/scitldr",
            "local_path": "domains/scientific/scitldr",
            "version": None,
            "columns_to_remove": ["source", "source_labels", "rouge_scores", "paper_id", "target"],
            "source": "hugging_face"
        }
    },
    SumDomains.MEDICAL: {
        "pubmed": {
            "dataset_id": "ccdv/pubmed-summarization",
            "local_path": "domains/medical/pubmed",
            "version": None,
            "columns_to_remove": ["article", "abstract"],
            "source": "hugging_face"
        },
        "cord19": {
            "dataset_id": "allenai/cord19",
            "local_path": "domains/medical/cord19",
            "version": "fulltext",
            "columns_to_remove": ['cord_uid', 'sha', 'source_x', 'title', 'doi', 'abstract', 'publish_time', 'authors',
                                  'journal', 'url', 'fulltext'],
            "source": "hugging_face"
        },
        "sci_lay": {
            "dataset_id": "paniniDot/sci_lay",
            "local_path": "domains/medical/sci_lay",
            "version": None,
            "columns_to_remove": ["article", "abstract"],
            "source": "hugging_face"
        },
        "mslr": {

        }
    },
    SumDomains.LEGAL: {
        "multi_lex": {
            "dataset_id": "allenai/multi_lexsum",
            "local_path": "domains/legal/multi_lex",
            "version": "v20230518",
            "columns_to_remove": ["id", "sources", "summary/long", "summary/short", "summary/tiny", "case_metadata",
                                  "sources_metadata"],
            "source": "hugging_face"
        },
        "eur_lex_sum": {

        },
        "bill_sum": {

        }
    },
    SumDomains.NEWS: {
        "cnn_dm": {
            "dataset_id": "ccdv/cnn_dailymail",
            "local_path": "domains/news/cnn_daily_mail",
            "version": "3.0.0",
            "columns_to_remove": ["id", "article", "highlights"],
            "source": "hugging_face"
        },
        "multi_news": {
            "dataset_id": "alexfabbri/multi_news",
            "local_path": "domains/news/multi_news",
            "version": None,
            "columns_to_remove": ["document"],
            "source": "hugging_face"
        },
        "x_sum": {
            "dataset_id": "EdinburghNLP/xsum",
            "local_path": "domains/news/x_sum",
            "version": None,
            "columns_to_remove": ["id", "document"],
            "source": "hugging_face"
        },
        "newsroom": {
            "dataset_id": "newsroom_extracted_data",  # "lil-lab/newsroom",
            "local_path": "domains/news/newsroom",
            "version": None,
            "columns_to_remove": ["compression", "compression_bin", "coverage", "coverage_bin", "date", "density",
                                  "density_bin", "text", "title", "url", "archive"],
            "source": ".tar",
            "download_url": "https://lil.nlp.cornell.edu/newsroom/download/index.html"
        }
    }
}


DEFAULT_DOMAIN_PROMPT = {
    SumDomains.SCIENTIFIC.name: """
        Provide a summary of the given scientific article that includes the following elements:

        1. Objective: What is the main research question or hypothesis?

        2. Background: What is the theoretical context or motivation for the research?

        3. Methods: Detail the approach, experiments, or simulations conducted.

        4. Key Findings: What are the principal results or contributions of the paper?

        5. Conclusions: What insights or conclusions do the authors derive from their research?

        6. Broader Impact: How does this research contribute to its field or influence future work?""".strip(),

    SumDomains.MEDICAL.name: """
        Summarize the given medical study by addressing the following key points:

        1. Objective: What is the primary research question or aim of the study?

        2. Background: What is the clinical context or rationale behind the research?

        3. Methods: Describe the study design, population involved, and methodologies employed.

        4. Key Findings: Highlight the most important results or discoveries of the study.

        5. Conclusions: What conclusions do the authors reach based on their findings?

        6. Clinical Implications: How might these findings influence clinical practice or patient outcomes?""".strip(),

    SumDomains.LEGAL.name: """
        Summarize the given legal case study by focusing on the following aspects:

        1. Case Background: What is the context and significance of the case?

        2. Legal Question: What are the primary legal issues or questions being addressed?

        3. Arguments: What are the main arguments presented by both sides?

        4. Rulings: What decisions were made by the court, and on what basis?

        5. Key Precedents: Are there important precedents cited that influence the case?

        6. Implications: What are the potential implications of the ruling on future cases or legal interpretations?
        """.strip(),

    SumDomains.NEWS.name: """
        Summarize the given news article by capturing the following key points:

        1. Main Event: What is the primary event or issue being reported?

        2. Context: What background information is necessary to understand the significance of the news?

        3. Key Details: What are the most critical facts or figures related to the story?

        4. Reactions: How have different stakeholders or the public responded to the event?

        5. Implications: What are the potential consequences or future developments related to this news?

        6. Closing Statement: What is the overarching message or takeaway from the article?""".strip()
}
