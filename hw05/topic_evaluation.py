from statistics import mean

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


class TopicEvaluator:
    METRICS = ("c_v", "c_npmi", "c_uci", "u_mass")

    def __init__(self, tokenized_documents):
        self.texts = tokenized_documents
        self.dictionary = Dictionary(tokenized_documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_documents]

    @staticmethod
    def topic_diversity(topics):
        words = [word for topic in topics for word in topic]
        return len(set(words)) / len(words)

    def evaluate(self, topics_by_model):
        scores = {
            name: {"diversity": self.topic_diversity(topics)}
            for name, topics in topics_by_model.items()
        }
        all_topics = [topic for topics in topics_by_model.values() for topic in topics]

        for metric in self.METRICS:
            model = CoherenceModel(
                topics=all_topics,
                texts=self.texts,
                corpus=self.corpus,
                dictionary=self.dictionary,
                coherence=metric,
                topn=10,
                processes=1,
            )
            topic_scores = model.get_coherence_per_topic()

            start = 0
            for name, topics in topics_by_model.items():
                end = start + len(topics)
                scores[name][metric] = float(mean(topic_scores[start:end]))
                start = end

        return scores