import os
import sys

# add the upper levels to the sys path for import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import paddle
import numpy as np

from data_utils import language_mapping

from paddlenlp.transformers import InfoXLMModel, InfoXLMTokenizer
from paddlenlp.datasets import load_dataset
from tqdm import tqdm
import math

tokenizer = InfoXLMTokenizer.from_pretrained("infoxlm-base")

model = InfoXLMModel.from_pretrained("infoxlm-base")

assert (
    math.abs(model.embeddings.word_embeddings.weight.std() - 0.1529) < 0.01,
    "The InfoXLM weight is not loaded correctly, the word embedding layer's weight should have a standard deviation of 0.1529"
)  # to see if the correct weight is loaded, should see 0.1529

model.eval()

__all__ = ["get_sentence_embedding", "SentenceRetrieval"]


def get_sentence_embedding(text):
    encoded_input = tokenizer([text])
    encoder_outputs, _ = model(paddle.to_tensor(encoded_input["input_ids"]),
                               output_hidden_states=True)
    emb = encoder_outputs[7].mean(axis=1)
    # detach to numpy
    result = emb.detach().squeeze(0).numpy()
    return result / np.linalg.norm(result)


class SentenceRetrieval(object):

    def __init__(self, languageA, languageB):
        self.languageA = languageA
        self.languageB = languageB
        self.sentencesA = []
        self.sentencesB = []
        self.embeddingsA = []
        self.embeddingsB = []

    def add_sentence(self, language, sentence):
        embedding = get_sentence_embedding(sentence)
        if language == self.languageA:
            self.sentencesA.append(sentence)
            self.embeddingsA.append(embedding)
        elif language == self.languageB:
            self.sentencesB.append(sentence)
            self.embeddingsB.append(embedding)
        else:
            raise ValueError(
                f"language must be either {self.languageA} or {self.languageB}")

    def stack(self):
        # convert embeddingsA and embeddingsB to numpy array
        embeddingsA = np.array(self.embeddingsA)
        embeddingsB = np.array(self.embeddingsB)
        return embeddingsA, embeddingsB

    def add_sentence_pair(self, sentenceA, sentenceB):
        self.add_sentence(self.languageA, sentenceA)
        self.add_sentence(self.languageB, sentenceB)

    def match_and_calculate(self, from_lang, to_lang):
        if from_lang == self.languageA and to_lang == self.languageB:
            _, embedding_targets = self.stack()
            embedding_lookup = self.embeddingsA
        elif from_lang == self.languageB and to_lang == self.languageA:
            embedding_targets, _ = self.stack()
            embedding_lookup = self.embeddingsB
        matched = []
        for idx, sentence_emb in enumerate(embedding_lookup):
            # cosine similarity
            sim = np.dot(sentence_emb, embedding_targets.T) / (
                np.linalg.norm(sentence_emb) *
                np.linalg.norm(embedding_targets))
            # get the index of the max value
            # print(np.argmax(sim))
            # import pdb; pdb.set_trace()
            matched.append(np.argmax(sim))
        # calculate accuracy
        matched = np.array(matched)
        correct = np.sum(matched == np.arange(len(matched)))
        accuracy = correct / len(matched)
        print(f"{from_lang} -> {to_lang} accuracy: {accuracy:.3f}")
        return accuracy

    def evaluate(self, logger=None):
        # match and calculate
        ab = self.match_and_calculate(self.languageA, self.languageB)
        ba = self.match_and_calculate(self.languageB, self.languageA)
        if logger is not None:
            logger.info(
                f"{self.languageA} -> {self.languageB} accuracy: {ab:.3f}")
            logger.info(
                f"{self.languageB} -> {self.languageA} accuracy: {ba:.3f}")
        else:
            print(f"average accuracy: {(ab+ba)/2:.3f}")
        return (ab + ba) / 2, ab, ba

    def load_dataset(self):
        # filenames = get_language_pair_filenames(self.languageA, self.languageB)
        split = f"{language_mapping[self.languageA]}-{language_mapping[self.languageB]}"
        dataset = load_dataset("tatoeba", split)
        for line in tqdm(list(dataset[split]), total=1000):
            lang = line["lang"]
            eng = line["eng"]
            self.add_sentence_pair(lang, eng)
