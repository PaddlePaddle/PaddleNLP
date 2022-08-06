from sentence_retrieval_utils import SentenceRetrieval
import logging
import os

# set format
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
# log to file
logger = logging.getLogger()
logger.addHandler(
    logging.FileHandler("cross_lingual_sentence_retrieval_log.txt"))

all_X_languages = [
    "ar",
    "bg",
    "de",
    "el",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

langB = "en"
X_TO_ENGS = []
ENGS_TO_X = []
for langA in all_X_languages:
    logger.info(f"Testing {langA}:en")
    sr = SentenceRetrieval(langA, langB)
    sr.load_dataset()
    avg, x_to_eng, eng_to_x = sr.evaluate(logger=logger)
    X_TO_ENGS.append(x_to_eng)
    ENGS_TO_X.append(eng_to_x)
    logger.info(f"Testing X = {langA} done")

import pandas as pd

dataframe = pd.DataFrame({
    "X": all_X_languages,
    "en->x": ENGS_TO_X,
    "x->en": X_TO_ENGS
})
dataframe.to_csv("sentence_retrieval_results.csv", index=False)
