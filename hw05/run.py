import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from document_vectorization import build_document_term_matrices
from text_preprocessing import TextPreprocessor
from topic_modeling import TopicModeler
from topic_evaluation import TopicEvaluator


def topic_details(result, method):
    document_scores = result.document_scores
    word_weights = result.topic_word_weights

    if method == "LSA":
        document_scores = np.abs(document_scores)
        word_weights = np.abs(word_weights)

    top_word_indices = np.argsort(-word_weights, axis=1, kind="stable")[:, :10]
    document_topics = document_scores.argmax(axis=1)
    word_topics = word_weights.argmax(axis=0)
    return top_word_indices, document_topics, word_topics


def print_subject_groups(subjects, document_topics, k):
    counts_by_subject = {subject: [0] * k for subject in sorted(set(subjects))}

    for subject, topic in zip(subjects, document_topics, strict=True):
        counts_by_subject[subject][topic] += 1

    headers = "".join(f"{'T' + str(i + 1):>5}" for i in range(k))
    print(f"{'Subject':<24}{headers}   Main topic(s)")
    majority_topics = []

    for subject, counts in counts_by_subject.items():
        best_count = max(counts)
        total = sum(counts)
        main_topics = "/".join(
            f"T{i + 1}" for i, count in enumerate(counts) if count == best_count
        )
        numbers = "".join(f"{count:>5}" for count in counts)
        print(f"{subject:<24}{numbers}   {main_topics} ({best_count}/{total})")

        if best_count > total / 2:
            majority_topics.append(counts.index(best_count))

    if len(set(majority_topics)) == len(counts_by_subject):
        print("Each subject has most of its articles in a different topic.")
    else:
        print("Some subjects are split across topics or share their main topic.")


def draw_scatter(ax, embeddings, groups, labels=None):
    coordinates = PCA(n_components=2).fit_transform(embeddings)
    groups = np.asarray(groups)

    for group in sorted(set(groups)):
        points = coordinates[groups == group]
        ax.scatter(points[:, 0], points[:, 1], label=group, s=42, alpha=0.8)

    if labels is not None:
        order = coordinates[:, 1].argsort()
        label_positions = np.linspace(0.06, 0.94, len(labels))

        for index, y in zip(order, label_positions, strict=True):
            ax.annotate(
                labels[index],
                xy=coordinates[index],
                xytext=(1.02, y),
                textcoords="axes fraction",
                va="center",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "0.65", "lw": 0.7},
            )

    ax.set(xlabel="PC 1", ylabel="PC 2")
    ax.margins(0.18)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)


def report_model(
    method, representation, result, articles, feature_names, plot_dir, details
):
    top_indices, document_topics, word_topics = details
    
    k = result.document_scores.shape[1]
    subjects = [article["subject"] for article in articles]

    heading = f"{method} on {representation}"
    print(f"\n{heading}\n{'-' * len(heading)}")
    print(f"1. Number of topics: {k}")

    print("\n2. Top 10 words per topic")
    top_words = [
        [feature_names[index] for index in indices]
        for indices in top_indices
    ]
    for topic, words in enumerate(top_words, start=1):
        print(f"Topic {topic}: {', '.join(words)}")

    print("\n3. Article assignments")
    for number, (article, topic) in enumerate(
        zip(articles, document_topics, strict=True), start=1
    ):
        print(f"{number:02d}. {article['title']} ({article['subject']}) -> Topic {topic + 1}")

    print("\n4. How the subjects are grouped")
    print_subject_groups(subjects, document_topics, k)

    word_indices = np.unique(top_indices[:, :5])
    word_labels = [feature_names[index] for index in word_indices]
    word_groups = [f"Topic {topic + 1}" for topic in word_topics[word_indices]]
    word_embeddings = result.topic_word_weights[:, word_indices].T

    fig, (doc_ax, word_ax) = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    fig.suptitle(f"{heading} (k={k})")
    doc_ax.set_title("Documents by original subject")
    word_ax.set_title("Top words by strongest topic")

    draw_scatter(doc_ax, result.document_scores, subjects)
    draw_scatter(word_ax, word_embeddings, word_groups, word_labels)

    representation_name = representation.lower().replace("-", "_")
    plot_path = plot_dir / f"{method.lower()}_{representation_name}_k{k}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n5. Scatter plots: {plot_path.name}")
    
    return top_words


def run_experiments(articles, matrices, feature_names, plot_dir, k):
    plot_dir.mkdir(parents=True, exist_ok=True)
    modeler = TopicModeler(k=k)
    methods = {
        "LSA": modeler.fit_lsa,
        "NMF": modeler.fit_nmf,
        "LDA": modeler.fit_lda,
    }
    runs = {}

    print(f"\nTopics found (k={k})")
    for representation, matrix in matrices.items():
        for method, fit in methods.items():
            result = fit(matrix)
            details = topic_details(result, method)
            runs[(method, representation)] = (result, details)

            print(f"\n{method} on {representation}")
            for topic, indices in enumerate(details[0], start=1):
                keywords = ", ".join(feature_names[i] for i in indices[:3])
                print(f"  Topic {topic}: {keywords}")

    print("\nDetailed results")
    topics_by_model = {}
    for (method, representation), (result, details) in runs.items():
        topics_by_model[(method, representation)] = report_model(
            method, representation, result, articles, feature_names, plot_dir, details
        )

    return topics_by_model


def print_and_save_evaluation(scores, k, folder):
    metrics = (*TopicEvaluator.METRICS, "diversity")
    rows = [["Model", "Representation", *metrics]]

    for (method, representation), values in scores.items():
        row = [method, representation, *(f"{values[m]:.6f}" for m in metrics)]
        rows.append(row)

    print(f"\n6. Topic evaluation (k={k})")
    for model, representation, *values in rows:
        columns = "".join(f"{value:>11}" for value in values)
        print(f"{model:<7}{representation:<17}{columns}")

    output_path = folder / f"topic_scores_k{k}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)

    print(f"\nScores saved to {output_path.name}.")

def choose_corpus(folder):
    files = sorted(folder.glob("wikipedia_articles*.json"))
    if not files:
        raise SystemExit("No Wikipedia article JSON files found.")

    default = 1
    print("\nWikipedia corpora")
    for number, path in enumerate(files, start=1):
        print(f"{number}. {path.name}")
        if path.name == "wikipedia_articles_k3.json":
            default = number

    try:
        choice = int(input(f"Choose a corpus [{default}]: ").strip() or default)
    except ValueError:
        raise SystemExit("Enter a file number from the list.") from None

    if not 1 <= choice <= len(files):
        raise SystemExit("Enter a file number from the list.")

    return files[choice - 1]

def main(k):
    folder = Path(__file__).resolve().parent
    corpus_path = choose_corpus(folder)
    articles = json.loads(corpus_path.read_text(encoding="utf-8"))
    print(f"Corpus: {corpus_path.name} ({len(articles)} articles)")

    preprocessor = TextPreprocessor()
    documents = [preprocessor.preprocess(article["text"]) for article in articles]
    bow, tfidf, feature_names = build_document_term_matrices(documents)

    matrices = {"Bag-of-Words": bow, "TF-IDF": tfidf}
    topics_by_model = run_experiments(
        articles, matrices, feature_names, folder / "topic_plots", k
    )

    print("\nComputing coherence scores...")
    evaluator = TopicEvaluator(documents)
    scores = evaluator.evaluate(topics_by_model)
    print_and_save_evaluation(scores, k, folder)