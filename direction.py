"""
Directionality Analysis Script

Analyzes the directionality of various languages (LTR vs RTL) using text corpora from
NLTK's Europarl and UDHR. It computes:
1. Gini Coefficient [1]
2. Entropy [2]

References:
[1] De Maio, F. G. (2007). Income inequality measures. Journal of Epidemiology & Community Health, 61(10), 849-852.
[2] Shannon, C. E. (1948). A mathematical theory of communication. The Bell System Technical Journal, 27(3), 379–423.
"""

import sys
import csv
from collections import Counter
from typing import List, Tuple, Dict

import nltk
from nltk.corpus import europarl_raw, udhr
import numpy as np
from scipy.stats import entropy


def download_nltk_corpora() -> None:
    """
    Ensure that the necessary NLTK corpora ('europarl_raw' and 'udhr') are downloaded.

    This function checks for the presence of the required corpora in NLTK's data directory.
    If missing, it downloads them.
    """
    required_corpora = ['europarl_raw', 'udhr']
    for corpus_name in required_corpora:
        try:
            nltk.data.find(f'corpora/{corpus_name}')
        except LookupError:
            print(f"Downloading '{corpus_name}' corpus...")
            nltk.download(corpus_name)


def get_available_languages() -> List[str]:
    """
    Retrieve a list of available languages in the Europarl corpus.

    This function inspects the europarl_raw module to identify all 
    language submodules that have a 'raw()' method.

    Returns:
        List[str]: A list of language identifiers (e.g., 'english', 'french').
    """
    return [
        lang for lang in dir(europarl_raw)
        if lang.islower() and hasattr(getattr(europarl_raw, lang), 'raw')
    ]


def calculate_gini_coefficient(freqs: List[int]) -> float:
    """
    Calculate the Gini coefficient for a list of frequencies.

    The Gini coefficient measures inequality among values of a frequency distribution.
    A higher Gini coefficient indicates greater inequality.

    Args:
        freqs (List[int]): A list of frequency counts.

    Returns:
        float: The calculated Gini coefficient.
    
    Reference:
        De Maio, F. G. (2007). "Income inequality measures." Journal of Epidemiology & Community Health, 61(10), 849-852.
    """
    sorted_freqs = np.sort(freqs)
    n = len(sorted_freqs)

    if n == 0:
        return 0.0

    cumulative_sum = np.cumsum(sorted_freqs)
    total = cumulative_sum[-1]

    if total == 0:
        return 0.0

    index = np.arange(1, n + 1)
    # Gini formula: (2 * Σ(i * x_i) / (n * total)) - (n + 1)/n
    gini = (2 * np.sum(index * sorted_freqs)) / (n * total) - (n + 1) / n
    return gini


def calculate_entropy_value(freqs: List[int]) -> float:
    """
    Calculate the Shannon entropy of a list of frequencies, using base-2.

    Shannon entropy measures the unpredictability in a distribution, with higher values 
    indicating more diversity or randomness.

    Args:
        freqs (List[int]): A list of frequency counts.

    Returns:
        float: The calculated entropy value.
    
    Reference:
        Shannon, C. E. (1948). "A mathematical theory of communication." 
        The Bell System Technical Journal, 27(3), 379-423.
    """
    if not freqs:
        return 0.0
    return entropy(freqs, base=2)


def extract_character_frequencies(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract initial and final character frequencies from the provided text.

    This function processes the text to extract the first and last characters of each
    word, excluding single-character words.

    Args:
        text (str): The input text to analyze.

    Returns:
        Tuple[List[str], List[str]]: Two lists containing initial and final characters respectively.
    """
    words = [word.strip() for word in text.split() if word]
    initial_chars = [word[0].lower() for word in words if len(word) > 1]
    final_chars = [word[-1].lower() for word in words if len(word) > 1]
    return initial_chars, final_chars


def analyze_directionality(text: str) -> Dict[str, float]:
    """
    Analyze the text to predict the direction of the writing system.

    This function computes various statistical measures (Gini coefficients, Shannon entropy)
    for the initial and final characters to infer whether the text is likely LTR or RTL.

    Args:
        text (str): The input text to analyze.

    Returns:
        Dict[str, float]: A dictionary containing Gini coefficients, entropy values, 
                          differences, and the predicted direction.
    """
    initial_chars, final_chars = extract_character_frequencies(text)
    if not initial_chars or not final_chars:
        raise ValueError("No valid words found in the text for analysis.")

    # Frequencies of first and last characters
    initial_freqs = Counter(initial_chars)
    final_freqs = Counter(final_chars)

    # Calculate Gini
    initial_gini = calculate_gini_coefficient(list(initial_freqs.values()))
    final_gini = calculate_gini_coefficient(list(final_freqs.values()))

    # Calculate Entropy
    initial_entropy = calculate_entropy_value(list(initial_freqs.values()))
    final_entropy = calculate_entropy_value(list(final_freqs.values()))

    # Differences
    gini_difference = initial_gini - final_gini
    entropy_difference = initial_entropy - final_entropy

    # Max possible entropy is log2 of the number of unique chars in both distributions
    unique_chars = set(initial_chars + final_chars)
    max_entropy = np.log2(len(unique_chars)) if unique_chars else 0.0

    # Normalize Gini difference (range assumption: -1 to 1 => map to [0,1])
    normalized_gini_diff = (gini_difference + 1) / 2

    # Normalize Entropy difference => map to [0,1] based on the max possible entropy
    normalized_entropy_diff = (entropy_difference + max_entropy) / (2 * max_entropy) if max_entropy else 0.0

    # Final combined score: if positive => LTR, negative => RTL
    combined_score = normalized_entropy_diff - normalized_gini_diff
    if combined_score > 0:
        likely_direction = "Left-to-Right"
    elif combined_score < 0:
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
        "Normalized Gini Difference": round(normalized_gini_diff, 4),
        "Normalized Entropy Difference": round(normalized_entropy_diff, 4),
        "Combined Score": round(combined_score, 4),
        "Likely Direction": likely_direction
    }


def analyze_texts_for_language(
    language_name: str,
    text_data: str,
    sample_size: int = None
) -> List[Dict[str, float]]:
    """
    Given a language name and raw text, analyze both the original and 
    reversed text and return directionality metrics.

    Args:
        language_name (str): The name or identifier of the language.
        text_data (str): The entire corpus text for the language.
        sample_size (int, optional): If given, truncate text to this size.

    Returns:
        List[Dict[str, float]]: A list of dictionaries with analysis results 
                                (normal text + reversed text).
    """
    results = []
    token_count = len(text_data.split())

    if sample_size is not None:
        text_data = text_data[:sample_size]

    reversed_text_data = text_data[::-1]

    # Normal text analysis
    normal_analysis = analyze_directionality(text_data)
    normal_analysis.update({
        "Language": language_name,
        "Sample Size": len(text_data),
        "Text Type": "Normal",
        "Token Count": token_count
    })
    results.append(normal_analysis)

    # Reversed text analysis
    reversed_analysis = analyze_directionality(reversed_text_data)
    reversed_analysis.update({
        "Language": language_name,
        "Sample Size": len(reversed_text_data),
        "Text Type": "Reversed",
        "Token Count": token_count
    })
    results.append(reversed_analysis)

    return results


def process_languages(
    languages: List[str],
    europarl_sample_size: int = None,
    udhr_sample_size: int = None
) -> List[Dict[str, float]]:
    """
    Process each language in Europarl and selected UDHR languages (e.g., Arabic, Hebrew).
    For each language, analyze both normal and reversed text.

    Args:
        languages (List[str]): List of language identifiers from Europarl.
        europarl_sample_size (int, optional): Truncate Europarl text to this size if given.
        udhr_sample_size (int, optional): Truncate UDHR text to this size if given.

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing analysis results.
    """
    results = []
    # Analyze Europarl languages
    for language in languages:
        print(f"Processing Language: {language.capitalize()}")
        try:
            corpus = getattr(europarl_raw, language)
            text_data = corpus.raw()
            analysis_for_lang = analyze_texts_for_language(
                language_name=language.capitalize(),
                text_data=text_data,
                sample_size=europarl_sample_size
            )
            results.extend(analysis_for_lang)
            print(f"Analysis for {language.capitalize()} completed.\n")
        except Exception as e:
            print(f"Error processing {language}: {e}\n")

    # Additional UDHR languages (RTL examples: Arabic, Hebrew)
    udhr_languages = [
        ("Arabic", "Arabic_Alarabia-Arabic"),
        ("Hebrew", "Hebrew_Ivrit-Hebrew")
    ]
    for udhr_language, fileid in udhr_languages:
        print(f"Processing UDHR Language: {udhr_language}")
        try:
            text_data = udhr.raw(fileids=fileid)
            analysis_for_lang = analyze_texts_for_language(
                language_name=udhr_language,
                text_data=text_data,
                sample_size=udhr_sample_size
            )
            results.extend(analysis_for_lang)
            print(f"Analysis for {udhr_language} completed.\n")
        except Exception as e:
            print(f"Error processing {udhr_language}: {e}\n")

    return results


def display_results(results: List[Dict[str, float]]) -> None:
    """
    Display the analysis results in a tabular (TSV) format on the standard output.

    Args:
        results (List[Dict[str, float]]): The list of analysis result dictionaries to display.
    """
    if not results:
        print("No results to display.")
        return

    print("\n--- Comprehensive Analysis Results ---\n")

    fieldnames = [
        "Language", "Sample Size", "Token Count", "Text Type",
        "Initial Gini", "Final Gini", "Initial Entropy", "Final Entropy",
        "Gini Difference", "Entropy Difference",
        "Normalized Gini Difference", "Normalized Entropy Difference",
        "Combined Score", "Likely Direction"
    ]

    csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter='\t')
    csv_writer.writeheader()
    for result in results:
        csv_writer.writerow(result)


def save_results_to_csv(results: List[Dict[str, float]], filename: str = 'directionality_results.csv') -> None:
    """
    Save the analysis results to a CSV file with UTF-8 encoding.

    Args:
        results (List[Dict[str, float]]): The list of analysis result dictionaries to save.
        filename (str, optional): The name of the output CSV file. Defaults to 'directionality_results.csv'.
    """
    fieldnames = [
        "Language", "Sample Size", "Token Count", "Text Type",
        "Initial Gini", "Final Gini", "Initial Entropy", "Final Entropy",
        "Gini Difference", "Entropy Difference",
        "Normalized Gini Difference", "Normalized Entropy Difference",
        "Combined Score", "Likely Direction"
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


def main() -> None:
    """
    Main function to orchestrate the directionality analysis workflow:

    1. Download required NLTK corpora if missing.
    2. Retrieve available Europarl languages.
    3. Analyze each language's text (normal and reversed) plus select UDHR texts.
    4. Display results in tabular format.
    5. Optionally save results to CSV.
    """
    download_nltk_corpora()
    languages = get_available_languages()
    if not languages:
        print("No languages found in 'europarl_raw' corpus.")
        sys.exit(1)

    print(f"Available languages: {languages}\n")
    all_results = process_languages(languages)

    # Display results in console
    display_results(all_results)

    # Optionally save to CSV
    save_choice = input("\nWould you like to save the results to 'directionality_results.csv'? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_results_to_csv(all_results)
    else:
        print("Results not saved.")


if __name__ == "__main__":
    main()