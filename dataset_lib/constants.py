from enum import Enum
from typing import Optional


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


class DatasetInfo:
    domain: SumDomains
    name: str
    dataset_id: str
    local_path: str
    streaming: bool = True
    trust_remote_code: bool = True
    version: Optional[str] = None
    columns_to_remove: list = []
    source: str
    download_url: Optional[str] = None

    def __str__(self):
        return f"{self.domain.name}/{self.name}"

    @property
    def local_storage_path(self):
        return self.local_path


class ArxivDataset(DatasetInfo):
    domain = SumDomains.SCIENTIFIC
    name = "arxiv"

    def __init__(self):
        super().__init__()
        self.dataset_id = "ccdv/arxiv-summarization"
        self.local_path = "domains/scientific/arxiv"
        self.streaming = True
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["article", "abstract"]
        self.source = "hugging_face"
        self.download_url = ""


class SSNDataset(DatasetInfo):
    domain = SumDomains.SCIENTIFIC
    name = "ssn"

    def __init__(self):
        # TODO:
        print("SSN dataset needs to be implemented")
        pass

        # raise NotImplementedError
        # super().__init__()
        # self.dataset_id = ""
        # self.local_path = "domains/scientific/ssn"
        # self.streaming = True
        # self.trust_remote_code = True
        # self.version = None
        # self.columns_to_remove = []
        # self.source = "hugging_face"
        # self.download_url = ""


class ElsevierDataset(DatasetInfo):
    domain = SumDomains.SCIENTIFIC
    name = "elsevier"

    def __init__(self):
        super().__init__()
        self.dataset_id = "orieg/elsevier-oa-cc-by"
        self.local_path = "domains/scientific/elsevier"
        self.streaming = False
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["title", "abstract", "subjareas", "keywords", "asjc", "body_text", "author_highlights"]
        self.source = "hugging_face"
        self.download_url = ""


class ScitldrDataset(DatasetInfo):
    domain = SumDomains.SCIENTIFIC
    name = "scitldr"

    def __init__(self):
        super().__init__()
        self.dataset_id = "allenai/scitldr"
        self.local_path = "domains/scientific/scitldr"
        self.streaming = True
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["source", "source_labels", "rouge_scores", "paper_id", "target"]
        self.source = "hugging_face"
        self.download_url = ""

class PubmedDataset(DatasetInfo):
    domain = SumDomains.MEDICAL
    name = "pubmed"

    def __init__(self):
        super().__init__()
        self.dataset_id = "ccdv/pubmed-summarization"
        self.local_path = "domains/medical/pubmed"
        self.streaming = True
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["article", "abstract"]
        self.source = "hugging_face"
        self.download_url = ""


class Cord19Dataset(DatasetInfo):
    domain = SumDomains.MEDICAL
    name = "cord19"

    def __init__(self):
        super().__init__()
        self.dataset_id = "allenai/cord19"
        self.local_path = "domains/medical/cord19"
        self.streaming = False
        self.trust_remote_code = True
        self.version = "fulltext"
        self.columns_to_remove = ["cord_uid", "sha", "source_x", "title", "doi", "abstract", "publish_time", "authors",
                                  "journal", "url", "fulltext"]
        self.source = "hugging_face"
        self.download_url = ""


class SciLayDataset(DatasetInfo):
    domain = SumDomains.MEDICAL
    name = "sci_lay"

    def __init__(self):
        super().__init__()
        self.dataset_id = "paniniDot/sci_lay"
        self.local_path = "domains/medical/sci_lay"
        self.streaming = True
        self.trust_remote_code = True
        self.version = "all"
        self.columns_to_remove = ["doi", "pmcid", "plain_text", "technical_text", "full_text", "journal",
                                  "topics", "keywords"]
        self.source = "hugging_face"
        self.download_url = ""


class MSLRDataset(DatasetInfo):
    domain = SumDomains.MEDICAL
    name = "mslr"

    def __init__(self):
        super().__init__()
        self.dataset_id = "allenai/mslr2022"
        self.local_path = "domains/medical/mslr"
        self.streaming = True
        self.trust_remote_code = True
        self.version = "ms2"
        self.columns_to_remove = ["doi", "pmcid", "plain_text", "technical_text", "full_text", "journal",
                                  "topics", "keywords"]
        self.source = "hugging_face"
        self.download_url = ""


class MultiLexDataset(DatasetInfo):
    domain = SumDomains.LEGAL
    name = "multi_lex"

    def __init__(self):
        super().__init__()
        self.dataset_id = "allenai/multi_lexsum"
        self.local_path = "domains/legal/multi_lex"
        self.streaming = True
        self.trust_remote_code = True
        self.version = "v20230518"
        self.columns_to_remove = ["id", "sources", "summary/long", "summary/short", "summary/tiny", "case_metadata",
                                  "sources_metadata"]
        self.source = "hugging_face"
        self.download_url = ""


class EurLexDataset(DatasetInfo):
    domain = SumDomains.LEGAL
    name = "eur_lex"

    def __init__(self):
        super().__init__()
        self.dataset_id = "dennlinger/eur-lex-sum"
        self.local_path = "domains/legal/eur_lex"
        self.streaming = True
        self.trust_remote_code = True
        self.version = "english"
        self.columns_to_remove = ["celex_id", "reference"]
        self.source = "hugging_face"
        self.download_url = ""


class BillSumDataset(DatasetInfo):
    domain = SumDomains.LEGAL
    name = "bill_sum"

    def __init__(self):
        super().__init__()
        self.dataset_id = "FiscalNote/billsum"
        self.local_path = "domains/legal/bill_sum"
        self.streaming = True
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["text", "title"]
        self.source = "hugging_face"
        self.download_url = ""


class CNNDMDataset(DatasetInfo):
    domain = SumDomains.NEWS
    name = "cnn_dm"

    def __init__(self):
        super().__init__()
        self.dataset_id = "ccdv/cnn_dailymail"
        self.local_path = "domains/news/cnn_daily_mail"
        self.streaming = False
        self.trust_remote_code = True
        self.version = "3.0.0"
        self.columns_to_remove = ["id", "article", "highlights"]
        self.source = "hugging_face"
        self.download_url = ""


class MultiNewsDataset(DatasetInfo):
    domain = SumDomains.NEWS
    name = "multi_news"

    def __init__(self):
        super().__init__()
        self.dataset_id = "alexfabbri/multi_news"
        self.local_path = "domains/news/multi_news"
        self.streaming = True
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["document"]
        self.source = "hugging_face"
        self.download_url = ""


class XSumDataset(DatasetInfo):
    domain = SumDomains.NEWS
    name = "x_sum"

    def __init__(self):
        super().__init__()
        self.dataset_id = "EdinburghNLP/xsum"
        self.local_path = "domains/news/x_sum"
        self.streaming = True
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["id", "document"]
        self.source = "hugging_face"
        self.download_url = ""


class NewsroomDataset(DatasetInfo):
    domain = SumDomains.NEWS
    name = "newsroom"

    def __init__(self):
        super().__init__()
        self.dataset_id = "newsroom_extracted_data"
        self.local_path = "domains/news/newsroom"
        self.streaming = True
        self.trust_remote_code = True
        self.version = None
        self.columns_to_remove = ["compression", "compression_bin", "coverage", "coverage_bin", "date", "density",
                                  "density_bin", "text", "title", "url", "archive"]
        self.source = ".tar"
        self.download_url = "https://lil.nlp.cornell.edu/newsroom/download/index.html"


datasets_info_dict = {
    SumDomains.SCIENTIFIC: {
        "arxiv": ArxivDataset(),
        "ssn": SSNDataset(),
        "elsevier": ElsevierDataset(),
        "scitldr": ScitldrDataset()
    },
    SumDomains.MEDICAL: {
        "pubmed": PubmedDataset(),
        "cord19": Cord19Dataset(),
        "sci_lay": SciLayDataset(),
        "mslr": MSLRDataset()
    },
    SumDomains.LEGAL: {
        "multi_lex": MultiLexDataset(),
        "eur_lex": EurLexDataset(),
        "bill_sum": BillSumDataset()
    },
    SumDomains.NEWS: {
        "cnn_dm": CNNDMDataset(),
        "multi_news": MultiNewsDataset(),
        "x_sum": XSumDataset(),
        "newsroom": NewsroomDataset()
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
