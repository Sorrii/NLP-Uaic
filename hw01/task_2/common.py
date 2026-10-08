from nltk.corpus import wordnet


def get_related_words(word):
    word = "_".join(word.lower().split())
    results = []

    for synset in wordnet.synsets(word):
        relations = {
            "Sinonime": set(),
            "Antonime": set(),
            "Hiperonime": set(),
            "Hiponime": set(),
            "Meronime": set(),
        }

        relation_pairs = {relation: set() for relation in relations}

        for lemma in synset.lemmas():
            if lemma.name().lower() != word:
                relations["Sinonime"].add(lemma.name().replace("_", " "))
                relation_pairs["Sinonime"].add((synset.name(), lemma.name().lower()))

            for antonym in lemma.antonyms():
                relations["Antonime"].add(antonym.name().replace("_", " "))
                relation_pairs["Antonime"].add((antonym.synset().name(), antonym.name().lower()))

        related_synsets = {
            "Hiperonime": synset.hypernyms(),
            "Hiponime": synset.hyponyms(),
            "Meronime": (
                synset.part_meronyms()
                + synset.member_meronyms()
                + synset.substance_meronyms()
            ),
        }

        for relation, synsets in related_synsets.items():
            for related_synset in synsets:
                for lemma in related_synset.lemmas():
                    relations[relation].add(lemma.name().replace("_", " "))
                    relation_pairs[relation].add((related_synset.name(), lemma.name().lower()))

        results.append({
            "synset": synset.name(),
            "part_of_speech": synset.pos(),
            "definition": synset.definition(),
            "relations": relations,
            "relation_pairs": relation_pairs,
        })

    return results