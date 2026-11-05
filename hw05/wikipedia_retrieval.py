import json
from pathlib import Path
from random import sample
from wikipediaapi import Wikipedia

ARTICLES_PER_SUBJECT = 10

SUBJECT_POOL = (
    "astronomy", "meteorology", "oceanography", "volcanology",
    "glaciology", "seismology", "paleontology", "hydrology",
    "mineralogy", "cartography", "ecology", "botany",
    "mycology", "marine biology", "entomology", "ornithology",
    "herpetology", "microbiology", "genetics", "immunology",
    "neuroscience", "epidemiology", "dentistry", "nutrition",
    "anatomy", "pharmacology", "veterinary medicine", "biochemistry",
    "organic chemistry", "materials science", "particle physics", "optics",
    "acoustics", "thermodynamics", "geometry", "number theory",
    "statistics", "cryptography", "robotics", "cybersecurity",
    "computer graphics", "telecommunications", "database systems",
    "operating systems", "renewable energy", "civil engineering",
    "aviation", "rail transport", "shipbuilding", "space exploration",
    "agriculture", "forestry", "beekeeping", "winemaking",
    "baking", "coffee", "cheese", "food preservation",
    "architecture", "landscape design", "urban planning", "interior design",
    "ceramics", "textile arts", "jewelry", "woodworking",
    "painting", "sculpture", "photography", "film history",
    "animation", "theatre", "ballet", "opera",
    "jazz", "classical music", "folk music", "musical instruments",
    "poetry", "science fiction", "literary criticism", "book publishing",
    "journalism", "advertising", "museum studies", "archaeology",
    "ancient Egypt", "ancient Greece", "medieval Europe", "Silk Road",
    "industrial revolution", "history of Japan", "Inca civilization",
    "anthropology", "sociology", "linguistics", "mythology",
    "philosophy", "ethics", "comparative religion", "education",
    "cognitive psychology", "economics", "international trade", "accounting",
    "public policy", "diplomacy", "constitutional law", "criminology",
    "demography", "public transportation", "human geography",
    "basketball", "tennis", "swimming", "cycling",
    "rock climbing", "skiing", "gymnastics", "sailing",
    "chess", "board games", "video game history",
)


def fetch_articles(
    k: int, subjects: list[str] | None = None
) -> list[dict[str, str]]:
    """Retrieve ten English Wikipedia articles per subject area."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")
    if isinstance(subjects, str):
        raise ValueError("Pass subject areas as a list of names.")

    subject_names = [subject.strip() for subject in (subjects or [])]
    if len(subject_names) > k or not all(subject_names):
        raise ValueError(f"Provide up to {k} non-empty subject names.")

    used_names = {subject.casefold() for subject in subject_names}
    if len(used_names) != len(subject_names):
        raise ValueError("Subject names must be distinct.")

    missing_count = k - len(subject_names)
    if missing_count:
        available_subjects = [
            subject for subject in SUBJECT_POOL
            if subject.casefold() not in used_names
        ]
        if missing_count > len(available_subjects):
            raise ValueError("The subject pool does not have enough distinct subjects.")
        subject_names.extend(sample(available_subjects, missing_count))

    wiki = Wikipedia(
        language="en",
        user_agent="NLP-UAIC-TopicModeling/1.0 (https://github.com/Sorrii/NLP-Uaic)",
    )
    
    articles = []
    seen_titles = set()

    for subject in subject_names:
        results = wiki.search(subject, limit=30)

        group = []
        for page in results.pages.values():
            text = page.text.strip()
            title = page.title
            if not text or title in seen_titles:
                continue

            group.append({"title": title, "text": text, "subject": subject})
            seen_titles.add(title)
            if len(group) == ARTICLES_PER_SUBJECT:
                break

        if len(group) < ARTICLES_PER_SUBJECT:
            raise RuntimeError(
                f"Found {len(group)} of {ARTICLES_PER_SUBJECT} "
                f"articles for '{subject}'."
            )

        articles.extend(group)

    return articles

def main():
    count_input = input("Number of subject areas [3]: ").strip()

    try:
        k = int(count_input or "3")
        names_input = input("Subject areas, separated by commas (optional): ")
        subjects = [name.strip() for name in names_input.split(",") if name.strip()]
        articles = fetch_articles(k, subjects)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from None

    output_path = Path(__file__).with_name(f"wikipedia_articles_k{k}.json")
    output_path.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(articles)} articles to {output_path.name}.")


if __name__ == "__main__":
    main()