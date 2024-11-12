import sys
import csv
import nltk
from nltk.corpus import europarl_raw
from collections import Counter
import numpy as np
from scipy.stats import entropy


def download_nltk_corpora():
    """
    Ensure that the necessary NLTK corpora are downloaded.
    """
    corpus_path = 'corpora/europarl_raw'
    try:
        nltk.data.find(corpus_path)
    except LookupError:
        print("Downloading 'europarl_raw' corpus...")
        nltk.download('europarl_raw')


def get_available_languages():
    """
    Retrieve a list of available languages in the europarl_raw corpus.
    Only include languages that are lowercase and have a callable 'raw' method.
    
    Returns:
        list: A list of language names as strings.
    """
    languages = []
    for lang in dir(europarl_raw):
        if lang.islower():
            corpus = getattr(europarl_raw, lang, None)
            if hasattr(corpus, 'raw') and callable(corpus.raw):
                languages.append(lang)
    return languages


def calculate_gini_coefficient(freqs):
    """
    Calculate the Gini coefficient for a list of frequencies.
    
    Args:
        freqs (list): A list of numerical frequencies.
        
    Returns:
        float: The Gini coefficient.
    """
    sorted_freqs = np.sort(freqs)  # Sort in ascending order
    n = len(sorted_freqs)
    if n == 0:
        return 0.0
    cumulative_sum = np.cumsum(sorted_freqs)
    total = cumulative_sum[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_freqs)) / (n * total) - (n + 1) / n
    return gini


def calculate_entropy_value(freqs):
    """
    Calculate the entropy based on the frequency of elements.
    
    Args:
        freqs (list): A list of numerical frequencies.
        
    Returns:
        float: The entropy value.
    """
    if not freqs:
        return 0.0
    return entropy(freqs, base=2)


def extract_character_frequencies(text):
    """
    Extract initial and final character frequencies from the provided text.
    
    Args:
        text (str): The input text to process.
        
    Returns:
        tuple: Two lists containing initial and final characters respectively.
    """
    words = [word.strip() for word in text.split() if word]
    initial_chars = [word[0].lower() for word in words if len(word) > 1]
    final_chars = [word[-1].lower() for word in words if len(word) > 1]
    return initial_chars, final_chars


def analyze_directionality(text):
    """
    Analyze the text to predict the direction of the writing system.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        dict: A dictionary containing analysis metrics and the likely direction.
    """
    initial_chars, final_chars = extract_character_frequencies(text)
    
    if not initial_chars or not final_chars:
        raise ValueError("No valid words found in the text for analysis.")
    
    initial_freqs = Counter(initial_chars)
    final_freqs = Counter(final_chars)
    
    if not initial_freqs or not final_freqs:
        raise ValueError("Insufficient character frequency data for analysis.")
    
    initial_gini = calculate_gini_coefficient(list(initial_freqs.values()))
    final_gini = calculate_gini_coefficient(list(final_freqs.values()))
    
    initial_entropy = calculate_entropy_value(list(initial_freqs.values()))
    final_entropy = calculate_entropy_value(list(final_freqs.values()))
    
    gini_difference = initial_gini - final_gini
    entropy_difference = initial_entropy - final_entropy
    
    # Determine likely direction based on sign of differences
    if gini_difference < 0 and entropy_difference > 0:
        likely_direction = "Left-to-Right"
    elif gini_difference > 0 and entropy_difference < 0:
        likely_direction = "Right-to-Left"
    else:
        likely_direction = "Indeterminate"
    
    return {
        "Initial Gini": round(initial_gini, 4),
        "Final Gini": round(final_gini, 4),
        "Initial Entropy": round(initial_entropy, 4),
        "Final Entropy": round(final_entropy, 4),
        "Gini Difference": round(gini_difference, 4),
        "Entropy Difference": round(entropy_difference, 4),
        "Likely Direction": likely_direction
    }


def process_languages(languages, sample_size=100000):
    """
    Process each language to analyze writing directionality.
    
    Args:
        languages (list): A list of language names as strings.
        sample_size (int, optional): Number of characters to sample from the corpus. Defaults to 1000.
        
    Returns:
        list: A list of dictionaries containing analysis results for each language.
    """
    results = []
    
    for language in languages:
        print(f"Processing Language: {language.capitalize()}")
        try:
            corpus = getattr(europarl_raw, language)
            text_data = corpus.raw()[:sample_size]
            print("Corpus loaded successfully.")
            
            analysis_results = analyze_directionality(text_data)
            analysis_results["Language"] = language.capitalize()
            analysis_results["Sample Size"] = sample_size  # Add sample size to results
            results.append(analysis_results)
            
            print("Analysis completed successfully.\n")
        
        except AttributeError:
            print(f"Language '{language}' not found in 'europarl_raw'. Skipping.\n")
        except ValueError as ve:
            print(f"Analysis Error: {ve}. Skipping.\n")
        except Exception as e:
            print(f"Unexpected Error: {e}. Skipping.\n")
    
    return results


def display_results(results):
    """
    Display the analysis results in a tabular format.
    
    Args:
        results (list): A list of dictionaries containing analysis results.
    """
    if not results:
        print("No results to display.")
        return
    
    print("\n--- Comprehensive Analysis Results ---\n")
    
    fieldnames = [
        "Language",
        "Sample Size",
        "Initial Gini",
        "Final Gini",
        "Initial Entropy",
        "Final Entropy",
        "Gini Difference",
        "Entropy Difference",
        "Likely Direction"
    ]
    
    # Initialize CSV writer with tab delimiter for better readability
    csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter='\t')
    csv_writer.writeheader()
    
    for result in results:
        csv_writer.writerow(result)


def save_results_to_csv(results, filename='directionality_results.csv'):
    """
    Save the analysis results to a CSV file.
    
    Args:
        results (list): A list of dictionaries containing analysis results.
        filename (str, optional): The filename for the CSV. Defaults to 'directionality_results.csv'.
    """
    if not results:
        print("No results to save.")
        return
    
    fieldnames = [
        "Language",
        "Sample Size",
        "Initial Gini",
        "Final Gini",
        "Initial Entropy",
        "Final Entropy",
        "Gini Difference",
        "Entropy Difference",
        "Likely Direction"
    ]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"Results successfully saved to '{filename}'.")
    except Exception as e:
        print(f"Error saving to CSV: {e}")


def main():
    """
    Main function to orchestrate the analysis of writing directionality.
    """
    # Step 1: Ensure necessary NLTK corpora are downloaded
    download_nltk_corpora()
    
    # Step 2: Retrieve all available languages
    available_languages = get_available_languages()
    if not available_languages:
        print("No languages found in 'europarl_raw' corpus.")
        sys.exit(1)
    
    print(f"Available languages in 'europarl_raw': {available_languages}\n")
    
    # Step 3: Analyze each language
    all_results = process_languages(available_languages)
    
    # Step 4: Display the results
    display_results(all_results)
    
    # Step 5: Prompt to save the results
    if all_results:
        save_choice = input("\nWould you like to save the results to 'directionality_results.csv'? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_results_to_csv(all_results)
        else:
            print("Results not saved.")


if __name__ == "__main__":
    main()