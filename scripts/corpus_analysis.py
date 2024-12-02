"""
Analyse the papyrus corpus dataset to extract insights and perform data cleaning.
Organization follows the given instructions:
4) Loading and cleaning the dataset
5) Corpus study: genre, location, and date
6) Greek text cleaning
7) Identifying person and place names
8) Study of spelling mistakes
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import torch
import difflib
from transformers import pipeline
from collections import defaultdict
from typing import List, Dict, Tuple
import difflib
from IPython.display import HTML, display
import pandas as pd


# 4) Chargement et nettoyage du dataset

def load_and_clean_dataset(file_path: str) -> pd.DataFrame:
    """Load and clean the dataset by removing unnecessary rows and columns."""
    # Charger le fichier CSV
    df = pd.read_csv(file_path)
    
    # Observer les 4 premières lignes
    print("Les 4 premières lignes du dataset :")
    print(df.head(4))
    
    # Suppression des lignes indésirables
    df = df[4:]  # Supposons que les 4 premières lignes soient des métadonnées
    
    # Remplacer 'nan' par des valeurs manquantes
    df = df.replace('nan', None)
    
    # Supprimer les textes manquants
    print("Nombre de textes qui n'ont pas été capturés pendant le scraping:", df['Full Text'].isnull().sum())
    df = df.dropna(subset=['Full Text'])
    
    # Compter les papyrus après nettoyage
    print("Nombre de papyrus après nettoyage:", len(df))
    
    # Trier la collection par ID
    df = df.sort_values(by='ID', ascending=True)
    
    return df


# 5) Étude de corpus : genre, lieu et date

def extract_genre(text: str) -> str:
    """Extract the first word from the content to identify the genre."""
    return text.split()[0].replace(":", "").replace("See", "").lower()

def genre_analysis(df: pd.DataFrame):
    """Analyze the genres in the dataset and display as a pie chart."""
    df['Genre'] = df['Content (beta!)'].apply(extract_genre)
    genre_counts = df['Genre'].value_counts()
    
    # Afficher un pie chart des genres
    plt.figure(figsize=(8, 8))
    plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
    plt.title("Distribution of Text Genres")
    plt.show()

def extract_city(text: str) -> str:
    """Extract the city name from the provenance information."""
    city = text.split("-")[0]
    return city.split("(")[0].replace("?", "").strip()

def provenance_analysis(df: pd.DataFrame):
    """Analyze the provenance of papyri and display as a bar plot."""
    df['Provenance'] = df['Provenance'].apply(extract_city)
    city_counts = df['Provenance'].value_counts()
    
    # Afficher un bar plot des provenances
    plt.figure(figsize=(10, 6))
    city_counts.plot(kind='bar', color='skyblue')
    plt.title("Provenance of Papyri")
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    # Conclusion
    print("Conclusion : la majorité des papyri proviennent d'Égypte, principalement de Aphrodito.")

def extract_date(text: str) -> str:
    """Extract the date information from the text."""
    return text.split(":")[-1].strip()

def cleaned_date(date_text: str) -> str:
    """Clean and format the date string to a standard format."""
    match = re.search(r'(AD|BC)\s*(\d+)(?:\s*[A-Za-z]+\s*\d*)?\s*-\s*(\d+)', date_text)
    if match:
        era, start_year, end_year = match.groups()
        return f"{era} {start_year} - {end_year}"
    
    match_single = re.search(r'(AD|BC)\s*(\d+)', date_text)
    if match_single:
        era, year = match_single.groups()
        return f"{era} {year}"
    
    return date_text

def extract_years(date_str: str):
    """Extract the years as a range, handling both AD and BC dates."""
    match_range_ad = re.search(r'AD (\d+)\s*-\s*(\d+)', date_str)
    match_single_ad = re.search(r'AD (\d+)', date_str)
    match_range_bc = re.search(r'BC (\d+)\s*-\s*(\d+)', date_str)
    match_single_bc = re.search(r'BC (\d+)', date_str)
    
    if match_range_ad:
        return range(int(match_range_ad.group(1)), int(match_range_ad.group(2)) + 1)
    elif match_single_ad:
        year = int(match_single_ad.group(1))
        return range(year, year + 1)
    elif match_range_bc:
        return range(-int(match_range_bc.group(1)), -int(match_range_bc.group(2)) - 1, -1)
    elif match_single_bc:
        year = -int(match_single_bc.group(1))
        return range(year, year + 1)
    
    return None

def date_analysis(df: pd.DataFrame):
    """Analyze the dates and display a density plot of the papyri over time."""
    df['Date'] = df['Date'].apply(extract_date).apply(cleaned_date)
    years = []
    for date in df['Date'].dropna():
        year_range = extract_years(date)
        if year_range:
            years.extend(year_range)
    
    # Générer le graphique de densité
    plt.figure(figsize=(10, 5))
    sns.kdeplot(years, fill=True)
    plt.title("Density of Papyrus per Years")
    plt.xlabel("Years")
    plt.ylabel("Density")
    plt.xlim(min(years), max(years)) 
    plt.grid()
    plt.show()

def extract_author(text: str) -> str:
    """Extract the author from the text."""
    if isinstance(text, str) and text.strip() != "":
        return text.split(",")[0].strip()
    return "nan"


# Text Cleaning Functions
def first_clean_text(text: str) -> str:
    """Clean the text by removing arabic numbers, lost lines, and special characters."""
    text = re.sub(r'\d+', '', text)  # Remove arabic numbers
    text = re.sub(r'\| gap \|', '', text)  # Remove lost lines
    text = text.replace("†", "").replace("⳨", "")  # Remove special characters
    return text

def uncertain_portion(text: str) -> float:
    """Calculate the proportion of uncertain characters in the text."""
    uncertain_chars = re.findall(r'\[.*?\]|\(.*?\)', text)
    uncertain_length = sum(len(uncertain) for uncertain in uncertain_chars)
    return uncertain_length / len(text)

def remove_brackets(text: str) -> str:
    """Remove brackets and parentheses while keeping their content."""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    return text

def normalize_text(text: str) -> str:
    """Normalize the text by removing diacritics and special characters."""
    text = text.lower()
    text = re.sub(r'[a-z]', '', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'\xa0', '', text)
    text = re.sub(r'\\', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'[\(\)]', '', text)
    # Remove special characters
    text = re.sub(r'[άὰᾶἀἁἂἃἄἅἆἇᾷᾳἄ]', 'α', text)
    text = re.sub(r'[έὲἐἑἒἓἔἕ]', 'ε', text)
    text = re.sub(r'[ίὶῖἰἱἲἳἴἵἶἷΐϊ]', 'ι', text)
    text = re.sub(r'[όὸὀὁὂὃὄὅ]', 'ο', text)
    text = re.sub(r'[ύὺῦὐὑὒὓὔὕὖὗ]', 'υ', text)
    text = re.sub(r'[ώὼῶὠὡὢὣὤὥὦὧῳὥῷᾧᾠ]', 'ω', text)
    text = re.sub(r'[ῆἠἡἢἣἤἥἦἧήἡῃὴῇ]', 'η', text)
    text = re.sub(r'[κ̣]', 'κ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_greek_text(text: str) -> str:
    """Nettoyer le texte en conservant les caractères grecs, en supprimant ponctuations et espaces superflus."""
    # Convertit en minuscules sans translittérer
    text = text.lower()
    # Supprime les chiffres et caractères non-alphabétiques grecs ou espaces
    text = re.sub(r'[^\s\wα-ωΑ-Ω]', '', text)
    # Normalisation des espaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Supprime les "____"
    text = re.sub(r'_+', '', text)

    return text

# NER Functions
def clean_people_list(text: str) -> str:
    """Clean the people list by removing special characters/sentences and replacing diacritics."""
    text = text.replace('Subscribe to export the table', '').replace('\\r\\n        \\t\\t\\t\\t\\tWe currently do not have any people attestations for this text.', '')
    text = normalize_text(text)

    # replace digital number
    text = re.sub(r'\d+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s,\'\"]', '', text)

    return text

def clean_places_list(text: str) -> str:
    """Clean the places list by removing special characters/sentences and replacing diacritics."""
    text = normalize_text(text)
    # replace digital number
    text = re.sub(r'\d+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s,\'\"]', '', text)

    return text

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
ner_pipeline = pipeline('ner', model='UGARIT/grc-ner-bert', device=device, aggregation_strategy = 'first')

def normalize_entity(text: str) -> str:
    """Normalise le texte en supprimant les accents, la ponctuation et les espaces superflus."""
    # Convertit en minuscules
    text = text.lower()
    # Retire la ponctuation et les espaces inutiles
    text = re.sub(r"[^\w\s]", "", text)  # Retire la ponctuation
    text = text.strip()  # Supprime les espaces en début et fin de chaîne
    return text

def extract_and_classify_entities(text, ner):
    entities = ner(text)
    people, places, other = [], [], []
    for entity in entities:
        if entity['entity_group'] == 'PER':
            people.append(entity['word'])
        elif entity['entity_group'] == 'LOC':
            places.append(entity['word'])
        else:
            other.append(entity['word'])
    return people, places, other


def parse_entities(entities_str: str) -> List[str]:
    """
    Parse une chaîne de caractères contenant des entités et retourne une liste d'entités.
    Gère les cas particuliers comme les chaînes vides ou '[]'.
    """
    # Si c'est déjà une liste, la retourner nettoyée
    if isinstance(entities_str, list):
        return [str(e).strip().strip('"') for e in entities_str if str(e).strip()]
        
    # Si c'est une valeur nulle ou une liste vide sous forme de chaîne
    if pd.isna(entities_str) or entities_str == '[]' or not entities_str:
        return []
    
    try:
        # Nettoyer les caractères spéciaux et diviser la chaîne
        entities = entities_str.strip('"[]').split(', ')
        # Nettoyer chaque entité et filtrer les chaînes vides
        return [e.strip().strip('"') for e in entities if e.strip()]
    except AttributeError:
        # En cas d'erreur, retourner une liste vide
        return []

def calculate_intersection(list1: List[str], list2: List[str]) -> int:
    """
    Calcule l'intersection entre deux listes.
    """
    return len([x for x in list1 if x in list2])

def calculate_metrics(reference: List[str], predicted: List[str]) -> Dict[str, float]:
    """
    Calcule la précision, le rappel et le F1 score pour une liste de prédictions.
    
    Args:
        reference: Liste des entités de référence
        predicted: Liste des entités prédites
    
    Returns:
        Dictionnaire contenant la précision, le rappel et le F1 score
    """
    true_positives = calculate_intersection(reference, predicted)
    false_positives = len([x for x in predicted if x not in reference])
    false_negatives = len([x for x in reference if x not in predicted])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_ner_scores(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calcule les F1 scores strict et tolérant pour l'ensemble du corpus.
    
    Args:
        df: DataFrame contenant les colonnes de référence et de prédiction
    
    Returns:
        Tuple contenant (F1 score strict, F1 score tolérant)
    """
    total_strict_f1 = 0
    total_tolerant_f1 = 0
    valid_docs = 0
    
    for _, row in df.iterrows():
        # Parser les entités de référence
        ref_persons = parse_entities(row['Personnes'])
        ref_places = parse_entities(row['Lieux'])
        
        # Parser les entités prédites
        pred_persons = parse_entities(row['People Ugarit'])
        pred_places = parse_entities(row['Places Ugarit'])
        pred_others = parse_entities(row['Other Ugarit'])
        
        # Calcul strict (en tenant compte des catégories)
        person_metrics = calculate_metrics(ref_persons, pred_persons)
        place_metrics = calculate_metrics(ref_places, pred_places)
        strict_f1 = (person_metrics['f1'] + place_metrics['f1']) / 2
        
        # Calcul tolérant (sans tenir compte des catégories)
        all_ref = ref_persons + ref_places
        all_pred = pred_persons + pred_places + pred_others
        tolerant_metrics = calculate_metrics(all_ref, all_pred)
        
        if pd.notna(strict_f1) and pd.notna(tolerant_metrics['f1']):
            total_strict_f1 += strict_f1
            total_tolerant_f1 += tolerant_metrics['f1']
            valid_docs += 1
    
    avg_strict_f1 = total_strict_f1 / valid_docs if valid_docs > 0 else 0
    avg_tolerant_f1 = total_tolerant_f1 / valid_docs if valid_docs > 0 else 0
    
    return avg_strict_f1, avg_tolerant_f1


# Sound Change Functions
def extract_sound_change(text):
    """Extract the sound change from the text irregularities."""
    changes = []
    if isinstance(text, str):
        # Split the text into individual changes
        entries = text.strip("[]").split(", ")
        for entry in entries:
            # print(entry)
            entry = normalize_text(entry)
            parts = entry.split(":")
            # print(parts)
            if len(parts) == 2:  # Ensure there are exactly two parts
                old = parts[0].replace("'", "").strip()
                new = parts[1].replace("read", "").replace("'", "").strip()
                changes.append((old, new))
    return changes

def process_text_irregularities(df, column_name='Text Irregularities'):
    """
    Process the DataFrame to extract sound changes from the specified column.
    """
    # Extract the sound changes
    sound_changes = []
    for irregularity in df[column_name].dropna():
        # print(irregularity)
        changes = extract_sound_change(irregularity)
        sound_changes.extend(changes)
    
    # Convert to DataFrame
    sound_change_df = pd.DataFrame(sound_changes, columns=['old', 'new'])
    sound_change_df = sound_change_df[sound_change_df['old'].str.strip() != '']
    sound_change_df = sound_change_df[sound_change_df['new'].str.strip() != '']

    sound_change_df.to_csv("result/sound_change_df.csv", index=False)

    return sound_change_df

def map_character_changes(old_word, new_word):
    """
    Map character changes between an old word and a new word.
    """
    diff = list(difflib.ndiff(old_word, new_word))
    changes = defaultdict(lambda: defaultdict(int))
    prev_char = None  # Initialize prev_char
    buffer_new_chars = []  # Buffer to accumulate new characters

    for d in diff:
        if d.startswith('- '):  # Character removed from the old word
            prev_char = d[2:]
        elif d.startswith('+ '):  # Character added in the new word
            char = d[2:]
            buffer_new_chars.append(char)
        elif d.startswith('  '):  # No change
            if prev_char and buffer_new_chars:
                # If the buffer length is within the allowed limit, register the change
                if len(buffer_new_chars) <= 3:
                    new_chars = ''.join(buffer_new_chars)
                    changes[prev_char][new_chars] += 1
                # Reset the buffer regardless of its length
                buffer_new_chars = []
            prev_char = d[2:]
        else:
            # Handle cases where the buffer is not empty when there's no match
            if buffer_new_chars and prev_char:
                if len(buffer_new_chars) <= 3:
                    new_chars = ''.join(buffer_new_chars)
                    changes[prev_char][new_chars] += 1
            buffer_new_chars = []
            prev_char = None

    # If the buffer is not empty at the end, process the last accumulated change
    if buffer_new_chars and prev_char:
        if len(buffer_new_chars) <= 3:
            new_chars = ''.join(buffer_new_chars)
            changes[prev_char][new_chars] += 1

    return changes

def aggregate_character_changes(dataframe):
    """
    Aggregate character changes across all word pairs in a dataframe.
    
    Parameters:
        dataframe (pd.DataFrame): The dataframe containing 'old' and 'new' word pairs.
        
    Returns:
        dict: A dictionary mapping characters to their changes and frequencies.
    """
    character_changes = defaultdict(lambda: defaultdict(int))
    for old_word, new_word in zip(dataframe['old'], dataframe['new']):
        changes = map_character_changes(old_word, new_word)
        for char, change_dict in changes.items():
            for new_char, count in change_dict.items():
                character_changes[char][new_char] += count

    most_changes = {k: dict(v) for k, v in character_changes.items() if sum(v.values()) > 30}
    sorted_changes = sorted(character_changes.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]
    most_10_frequent_change_sound = {k: dict(v) for k, v in sorted_changes}
    return most_changes, most_10_frequent_change_sound

def print_differences(old: str, new: str) -> str:
    """
    Create a string showing differences between old and new strings using ASCII markers:
    - [-] for deleted characters
    - [+] for added characters
    - No marker for unchanged characters
    """
    diff = difflib.SequenceMatcher(None, old, new)
    result = []
    
    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        if tag == 'replace':
            result.append(f'[-{old[i1:i2]}+{new[j1:j2]}]')
        elif tag == 'delete':
            result.append(f'[-{old[i1:i2]}]')
        elif tag == 'insert':
            result.append(f'[+{new[j1:j2]}]')
        elif tag == 'equal':
            result.append(old[i1:i2])
    
    return ''.join(result)

def display_text_differences(sound_change_df: pd.DataFrame, n: int = 10) -> None:
    """
    Display the first n rows of sound changes with text-based difference markers.
    
    Parameters:
        sound_change_df: DataFrame containing 'old' and 'new' columns
        n: Number of rows to display (default: 10)
    """
    print("\nSound Changes Analysis:")
    print("=" * 60)
    print(f"{'Original':<20} {'Changes':<40}")
    print("-" * 60)
    
    for _, row in sound_change_df.head(n).iterrows():
        diff_text = print_differences(row['old'], row['new'])
        print(f"{row['old']:<20} {diff_text:<40}")
    
    print("=" * 60)
    print("\nLegend:")
    print("[-x] : Deleted characters")
    print("[+x] : Added characters")
    print("Plain text : Unchanged characters")

def plot_top_sound_changes_pie_charts(character_changes, top_n=8):
    """
    Plot pie charts for the top N most frequently modified characters.
    """
    import matplotlib.pyplot as plt

    # Sort the characters by the total number of changes and select the top N
    sorted_characters = sorted(character_changes.items(), key=lambda x: sum(x[1].values()), reverse=True)[:top_n]
    
    # Number of pie charts to plot (one for each character)
    num_plots = len(sorted_characters)
    fig, axes = plt.subplots((num_plots + 1) // 2, 2, figsize=(14, num_plots * 3))
    fig.subplots_adjust(hspace=0.5)
    axes = axes.flatten()

    for i, (char, changes) in enumerate(sorted_characters):
        # Prepare data for the pie chart
        labels = list(changes.keys())
        sizes = list(changes.values())

        # Plot pie chart
        ax = axes[i]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title(f"Changes for '{char}'")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()

def new_dataframe(df_cleaned):
    """
    Create a new DataFrame from the cleaned DataFrame.
    """
    # Select the relevant columns
    new_df = df_cleaned[['ID', 'Author', 'Genre', 'Date', 'Cleaned Text', 'Provenance',  'Personnes', 'Lieux', 'Text Irregularities']]
    new_df.to_csv("result/clean_papyrus-corpus.csv", index=False)

    return new_df

# Main Program Execution
def main():
    """
    Main execution function organized according to the instructions.
    """
    file_path = "csv/papyrus_corpus.csv"
    
    print("\n=== 4) Loading and cleaning the dataset ===")
    # Load and clean the dataset
    df = load_and_clean_dataset(file_path)

    # Extract authors right after loading
    df['Author'] = df['Authors / works'].apply(extract_author)
    
    print("\n=== 5) Corpus study: genre, location, and date ===")
    # Genre analysis with pie chart
    print("\nAnalyzing genres...")
    genre_analysis(df)
    
    # Provenance analysis with bar plot
    print("\nAnalyzing provenances...")
    provenance_analysis(df)
    
    # Date analysis with density plot
    print("\nAnalyzing dates...")
    date_analysis(df)

    print("\n=== 6) Greek text cleaning ===")
    # Clean text by removing numbers, lost lines, and special characters
    print("\nCleaning text...")
    df['Cleaned Text'] = df['Full Text'].apply(first_clean_text)

    # Calculate uncertain portion
    print("\nCalculating uncertain portions...")
    df['Uncertain Portion'] = df['Cleaned Text'].apply(uncertain_portion)
    uncertain_papyri = df[df['Uncertain Portion'] > 0.33]
    print(f"Number of papyri with more than a third of uncertain text: {len(uncertain_papyri)}")
    
    # Remove brackets while keeping content
    print("\nRemoving brackets while keeping content...")
    df['Cleaned Text'] = df['Cleaned Text'].apply(remove_brackets)
    
    # Final text cleaning
    print("\nPerforming final text cleaning...")
    df['Cleaned Text'] = df['Cleaned Text'].apply(clean_greek_text)

    print("\n=== 7) Identifying person and place names ===")
    # Clean people and places lists
    print("\nCleaning people and places lists...")
    df['Personnes'] = df['People List'].apply(clean_people_list)
    df['Lieux'] = df['Places List'].apply(clean_places_list)

    # Configure and use NER model
    print("\nApplying NER model...")
    df[['People Ugarit', 'Places Ugarit', 'Other Ugarit']] = df['Cleaned Text'].apply(
        lambda x: pd.Series(extract_and_classify_entities(x, ner_pipeline))
    )

    # Clean extracted entities
    print("\nCleaning extracted entities...")
    df['Personnes'] = df['Personnes'].apply(lambda x: ', '.join([clean_greek_text(name) for name in x.split(',')]))
    df['Lieux'] = df['Lieux'].apply(lambda x: ', '.join([clean_greek_text(name) for name in x.split(',')]))
    df['People Ugarit'] = df['People Ugarit'].apply(lambda x: ', '.join([normalize_entity(name) for name in x]) if isinstance(x, list) else x)
    df['Places Ugarit'] = df['Places Ugarit'].apply(lambda x: ', '.join([normalize_entity(name) for name in x]) if isinstance(x, list) else x)
    df['Other Ugarit'] = df['Other Ugarit'].apply(lambda x: ', '.join([normalize_entity(name) for name in x]) if isinstance(x, list) else x)
    
    # Evaluate NER system
    print("\nEvaluating NER system...")
    strict_f1, tolerant_f1 = calculate_ner_scores(df)
    print(f"NER System Evaluation Results:")
    print(f"Strict F1 Score (with categories): {strict_f1:.2%}")
    print(f"Tolerant F1 Score (without categories): {tolerant_f1:.2%}")

    print("\n=== 8) Study of spelling mistakes ===")
    # Process text irregularities
    print("\nProcessing text irregularities...")
    df['Text Irregularities'] = df['Text Irregularities'].apply(normalize_text)
    sound_change_df = process_text_irregularities(df)

    # Analyze sound changes
    print("\nAnalyzing sound changes...")
    most_changes, most_10 = aggregate_character_changes(sound_change_df)
    print("\nMost frequent sound changes:", most_10)
    
    # Display colored differences
    print("\nDisplaying colored differences for first 10 sound changes...")
    display_text_differences(sound_change_df)

    # Plot sound changes
    print("\nPlotting sound changes...")
    plot_top_sound_changes_pie_charts(most_changes)

    print("\n=== Creating clean dataset ===")
    # Create and save clean dataset
    new_df = new_dataframe(df)
    print("\nSaving complete dataset...")
    print("Analysis complete!")


# Run the main function
if __name__ == "__main__":
    main()