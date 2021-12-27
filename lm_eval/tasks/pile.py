import os

import lm_dataformat
import abc
import numpy as np
from lm_eval.base import rf, PerplexityTask
from ..metrics import mean, matthews_corrcoef, f1_score
from ..utils import general_detokenize
from best_download import download_file
from .common import HFTask


class PilePerplexityTask(PerplexityTask, HFTask):
    VERSION = 0
    DATASET_PATH = "the_pile"

    def download(self):
        super().download()
        self.data["validation"] = [x["text"] for x in self.data["train"]]

    def has_validation_docs(self):
        return True

class PileArxiv(PilePerplexityTask):
    PILE_SET_NAME = "ArXiv"


class PileBooks3(PilePerplexityTask):
    PILE_SET_NAME = "Books3"


class PileBookCorpus2(PilePerplexityTask):
    PILE_SET_NAME = "BookCorpus2"


class PileDmMathematics(PilePerplexityTask):
    PILE_SET_NAME = "DM Mathematics"


class PileEnron(PilePerplexityTask):
    DATASET_NAME = "enron_emails"


class PileEuroparl(PilePerplexityTask):
    PILE_SET_NAME = "EuroParl"
    DATASET_NAME = "europarl"


class PileFreeLaw(PilePerplexityTask):
    PILE_SET_NAME = "FreeLaw"
    DATASET_NAME = "free_law"


class PileGithub(PilePerplexityTask):
    PILE_SET_NAME = "Github"


class PileGutenberg(PilePerplexityTask):
    PILE_SET_NAME = "Gutenberg (PG-19)"


class PileHackernews(PilePerplexityTask):
    PILE_SET_NAME = "HackerNews"
    DATASET_NAME = "hacker_news"


class PileNIHExporter(PilePerplexityTask):
    PILE_SET_NAME = "NIH ExPorter"
    DATASET_NAME = "nih_exporter"


class PileOpenSubtitles(PilePerplexityTask):
    PILE_SET_NAME = "OpenSubtitles"


class PileOpenWebText2(PilePerplexityTask):
    PILE_SET_NAME = "OpenWebText2"


class PilePhilPapers(PilePerplexityTask):
    PILE_SET_NAME = "PhilPapers"


class PilePileCc(PilePerplexityTask):
    PILE_SET_NAME = "Pile-CC"


class PilePubmedAbstracts(PilePerplexityTask):
    PILE_SET_NAME = "PubMed Abstracts"
    DATASET_NAME = "pubmed"


class PilePubmedCentral(PilePerplexityTask):
    PILE_SET_NAME = "PubMed Central"
    DATASET_NAME = "pubmed_central"


class PileStackExchange(PilePerplexityTask):
    PILE_SET_NAME = "StackExchange"


class PileUspto(PilePerplexityTask):
    PILE_SET_NAME = "USPTO Backgrounds"
    DATASET_NAME = "uspto"


class PileUbuntuIrc(PilePerplexityTask):
    PILE_SET_NAME = "Ubuntu IRC"


class PileWikipedia(PilePerplexityTask):
    PILE_SET_NAME = "Wikipedia (en)"


class PileYoutubeSubtitles(PilePerplexityTask):
    PILE_SET_NAME = "YoutubeSubtitles"
