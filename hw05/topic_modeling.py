from dataclasses import dataclass

from numpy import ndarray
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD


@dataclass
class TopicModelResult:
    document_scores: ndarray
    topic_word_weights: ndarray


class TopicModeler:
    def __init__(self, k: int = 3, random_state: int = 42) -> None:
        self.k = k
        self.random_state = random_state

    def fit_lsa(self, matrix) -> TopicModelResult:
        model = TruncatedSVD(n_components=self.k, random_state=self.random_state)
        return self._fit(model, matrix)

    def fit_nmf(self, matrix) -> TopicModelResult:
        model = NMF(
            n_components=self.k,
            max_iter=1000,
            random_state=self.random_state,
        )
        return self._fit(model, matrix)

    def fit_lda(self, matrix) -> TopicModelResult:
        model = LatentDirichletAllocation(
            n_components=self.k,
            max_iter=20,
            random_state=self.random_state,
        )
        return self._fit(model, matrix)

    def _fit(self, model, matrix) -> TopicModelResult:
        if not 1 <= self.k <= min(matrix.shape):
            raise ValueError("k must be between 1 and the smaller matrix dimension.")

        document_scores = model.fit_transform(matrix)
        return TopicModelResult(
            document_scores=document_scores,
            topic_word_weights=model.components_,
        )