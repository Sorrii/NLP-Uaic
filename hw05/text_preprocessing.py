from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

WORDNET_POS = {"N": "n", "V": "v", "J": "a", "R": "r"}


class TextPreprocessor:
    def __init__(self) -> None:
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words("english"))

    @staticmethod
    def lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.replace("’", "'").replace("-", " ")
        return word_tokenize(text)

    def lemmatize(self, tokens: list[str]) -> list[str]:
        lemmas = []

        for token, tag in pos_tag(tokens):
            if not token.isalpha():
                continue

            wordnet_pos = WORDNET_POS.get(tag[0])
            lemma = (
                self._lemmatizer.lemmatize(token, pos=wordnet_pos)
                if wordnet_pos is not None
                else token
            )
            lemmas.append(lemma)

        return lemmas

    def remove_stop_words(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in self._stop_words]

    def preprocess(self, text: str) -> list[str]:
        lowercased_text = self.lowercase(text)
        tokens = self.tokenize(lowercased_text)
        lemmas = self.lemmatize(tokens)
        return self.remove_stop_words(lemmas)