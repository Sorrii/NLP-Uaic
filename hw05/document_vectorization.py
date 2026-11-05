from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _use_preprocessed_tokens(tokens: list[str]) -> list[str]:
    return tokens


def build_document_term_matrices(tokenized_documents: list[list[str]]):
    """Build Bag-of-Words and TF-IDF matrices with the same vocabulary."""
    bow_vectorizer = CountVectorizer(analyzer=_use_preprocessed_tokens)
    bow_matrix = bow_vectorizer.fit_transform(tokenized_documents)

    tfidf_vectorizer = TfidfVectorizer(
        analyzer=_use_preprocessed_tokens,
        vocabulary=bow_vectorizer.vocabulary_,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_documents)

    feature_names = bow_vectorizer.get_feature_names_out().tolist()
    return bow_matrix, tfidf_matrix, feature_names