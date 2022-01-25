import numpy as np
from nlpaug.util import Action, Doc, PartOfSpeech
from nlpaug.augmenter.word import SynonymAug


class GeometricSynonymAug(SynonymAug):
    # https://arxiv.org/abs/1509.01626.pdf
    """
    Augmenter that leverage semantic meaning to substitute word, using geometric distribution to sample.
    """
    def _get_geometric_num(self, max_num=None, p=0.5):
        assert max_num is None or max_num > 0
        num = np.random.geometric(p)
        while max_num and num > max_num:
            num = np.random.geometric(p)
        return num

    def generate_aug_cnt(self, size, aug_p=None):
        if size == 0:
            return 0
        return self._get_geometric_num(max_num=size)

    def geometric_sample(self, word_list: list, original_token: str) -> str:
        """
        sample a word from word_list as in "Character-level Convolutional Networks for Text Classification"
        :param word_list: list[str]
        :return: sampled word: str
        """
        word_list = list(set(word_list))
        """every synonym to a word or phrase is ranked by the semantic closeness to the most frequently seen meaning"""
        original_meaning = self.model.model.synsets(original_token)[0]
        closeness = {}
        for word in word_list:
            closeness[word] = self.model.model.synsets(word)[0].wup_similarity(
                original_meaning)
        word_list = sorted(word_list, key=lambda x: closeness[x], reverse=True)
        """The index s of the synonym chosen given a word is also determined by a another geometric distribution"""
        s = self._get_geometric_num(len(word_list))
        return word_list[s - 1]

    def substitute(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0
        doc = Doc(data, self.tokenizer(data))

        original_tokens = doc.get_original_tokens()

        pos = self.model.pos_tag(original_tokens)

        aug_idxes = self._get_aug_idxes(pos)
        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data

        for aug_idx in aug_idxes:
            original_token = original_tokens[aug_idx]

            word_poses = PartOfSpeech.constituent2pos(pos[aug_idx][1])
            candidates = []
            if word_poses is None or len(word_poses) == 0:
                # Use every possible words as the mapping does not defined correctly
                candidates.extend(self.model.predict(pos[aug_idx][0]))
            else:
                for word_pos in word_poses:
                    candidates.extend(
                        self.model.predict(pos[aug_idx][0], pos=word_pos))

            candidates = [
                c for c in candidates if c.lower() != original_token.lower()
            ]

            if len(candidates) > 0:
                # candidate = self.sample(candidates, 1)[0]
                candidate = self.geometric_sample(
                    candidates,
                    original_token=pos[aug_idx][0])  # the only line changed
                candidate = candidate.replace("_", " ").replace("-",
                                                                " ").lower()
                substitute_token = self.align_capitalization(
                    original_token, candidate)

                if aug_idx == 0:
                    substitute_token = self.align_capitalization(
                        original_token, substitute_token)

                change_seq += 1
                doc.add_change_log(aug_idx,
                                   new_token=substitute_token,
                                   action=Action.SUBSTITUTE,
                                   change_seq=self.parent_change_seq +
                                   change_seq)

        if self.include_detail:
            return self.reverse_tokenizer(
                doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return self.reverse_tokenizer(doc.get_augmented_tokens())
