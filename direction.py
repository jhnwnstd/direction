import sys
import csv
import nltk
from nltk.corpus import europarl_raw, udhr
from collections import Counter
import numpy as np
from scipy.stats import entropy


def download_nltk_corpora():
    """Ensure that the necessary NLTK corpora are downloaded."""
    for corpus_name in ['europarl_raw', 'udhr']:
        try:
            nltk.data.find(f'corpora/{corpus_name}')
        except LookupError:
            print(f"Downloading '{corpus_name}' corpus...")
            nltk.download(corpus_name)


def get_available_languages():
    """Retrieve a list of available languages in the europarl_raw corpus."""
    return [lang for lang in dir(europarl_raw) if lang.islower() and hasattr(getattr(europarl_raw, lang), 'raw')]


def calculate_gini_coefficient(freqs):
    """Calculate the Gini coefficient for a list of frequencies."""
    sorted_freqs = np.sort(freqs)
    n = len(sorted_freqs)
    if n == 0:
        return 0.0
    cumulative_sum = np.cumsum(sorted_freqs)
    total = cumulative_sum[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_freqs)) / (n * total) - (n + 1) / n


def calculate_entropy_value(freqs):
    """Calculate entropy based on the frequency of elements."""
    if not freqs:
        return 0.0
    return entropy(freqs, base=2)


def extract_character_frequencies(text):
    """Extract initial and final character frequencies from the provided text."""
    words = [word.strip() for word in text.split() if word]
    initial_chars = [word[0].lower() for word in words if len(word) > 1]
    final_chars = [word[-1].lower() for word in words if len(word) > 1]
    return initial_chars, final_chars


def analyze_directionality(text):
    """Analyze the text to predict the direction of the writing system."""
    initial_chars, final_chars = extract_character_frequencies(text)
    
    if not initial_chars or not final_chars:
        raise ValueError("No valid words found in the text for analysis.")
    
    # Token count for normalization
    token_count = len(text.split())
    if token_count == 0:
        raise ValueError("Token count is zero; cannot normalize.")
    
    # Frequency counts
    initial_freqs = Counter(initial_chars)
    final_freqs = Counter(final_chars)
    
    # Raw Gini and Entropy
    initial_gini = calculate_gini_coefficient(list(initial_freqs.values()))
    final_gini = calculate_gini_coefficient(list(final_freqs.values()))
    initial_entropy = calculate_entropy_value(list(initial_freqs.values()))
    final_entropy = calculate_entropy_value(list(final_freqs.values()))
    
    # Normalized Gini and Entropy
    normalized_initial_gini = initial_gini / token_count
    normalized_final_gini = final_gini / token_count
    normalized_initial_entropy = initial_entropy / token_count
    normalized_final_entropy = final_entropy / token_count
    
    # Differences using normalized values
    gini_difference = normalized_initial_gini - normalized_final_gini
    entropy_difference = normalized_initial_entropy - normalized_final_entropy
    
    # Combined score
    combined_score = entropy_difference - gini_difference
    
    # Determine likely direction
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
        "Normalized Initial Gini": round(normalized_initial_gini, 8),
        "Normalized Final Gini": round(normalized_final_gini, 8),
        "Normalized Initial Entropy": round(normalized_initial_entropy, 8),
        "Normalized Final Entropy": round(normalized_final_entropy, 8),
        "Gini Difference": round(gini_difference, 8),
        "Entropy Difference": round(entropy_difference, 8),
        "Combined Score": round(combined_score, 8),
        "Likely Direction": likely_direction
    }


def process_languages(languages, europarl_sample_size=None, udhr_sample_size=None):
    """Process each language in Europarl and UDHR, with reversed text testing."""
    results = []

    for language in languages:
        print(f"Processing Language: {language.capitalize()}")
        try:
            corpus = getattr(europarl_raw, language)
            text_data = corpus.raw()
            token_count = len(text_data.split())  # Count tokens in the corpus

            # If sample size is specified, truncate the text
            if europarl_sample_size is not None:
                text_data = text_data[:europarl_sample_size]

            reversed_text_data = text_data[::-1]

            # Analyze normal text
            normal_results = analyze_directionality(text_data)
            normal_results.update({
                "Language": language.capitalize(),
                "Sample Size": len(text_data) if europarl_sample_size is None else europarl_sample_size,
                "Text Type": "Normal",
                "Token Count": token_count  # Record token count
            })
            results.append(normal_results)

            # Analyze reversed text
            reversed_results = analyze_directionality(reversed_text_data)
            reversed_results.update({
                "Language": language.capitalize(),
                "Sample Size": len(reversed_text_data) if europarl_sample_size is None else europarl_sample_size,
                "Text Type": "Reversed",
                "Token Count": token_count  # Record token count
            })
            results.append(reversed_results)

            print(f"Analysis for {language.capitalize()} completed.\n")

        except Exception as e:
            print(f"Error processing {language}: {e}\n")

    for udhr_language, fileid in [('Arabic', 'Arabic_Alarabia-Arabic'), ('Hebrew', 'Hebrew_Ivrit-Hebrew')]:
        print(f"Processing UDHR Language: {udhr_language}")
        try:
            text_data = udhr.raw(fileids=fileid)
            token_count = len(text_data.split())  # Count tokens in the UDHR text

            # If sample size is specified, truncate the text
            if udhr_sample_size is not None:
                text_data = text_data[:udhr_sample_size]

            reversed_text_data = text_data[::-1]

            # Analyze normal text
            normal_results = analyze_directionality(text_data)
            normal_results.update({
                "Language": udhr_language,
                "Sample Size": len(text_data) if udhr_sample_size is None else udhr_sample_size,
                "Text Type": "Normal",
                "Token Count": token_count  # Record token count
            })
            results.append(normal_results)

            # Analyze reversed text
            reversed_results = analyze_directionality(reversed_text_data)
            reversed_results.update({
                "Language": udhr_language,
                "Sample Size": len(reversed_text_data) if udhr_sample_size is None else udhr_sample_size,
                "Text Type": "Reversed",
                "Token Count": token_count  # Record token count
            })
            results.append(reversed_results)

            print(f"Analysis for {udhr_language} completed.\n")

        except Exception as e:
            print(f"Error processing {udhr_language}: {e}\n")

    return results


def display_results(results):
    """Display the analysis results in a tabular format."""
    if not results:
        print("No results to display.")
        return
    
    print("\n--- Comprehensive Analysis Results ---\n")
    
    # Updated fieldnames to include "Token Count"
    fieldnames = [
        "Language",
        "Sample Size",
        "Token Count",  # Added this field
        "Text Type",
        "Initial Gini",
        "Final Gini",
        "Initial Entropy",
        "Final Entropy",
        "Gini Difference",
        "Entropy Difference",
        "Combined Score",
        "Likely Direction"
    ]
    
    csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter='\t')
    csv_writer.writeheader()
    
    for result in results:
        csv_writer.writerow(result)



def save_results_to_csv(results, filename='directionality_results.csv'):
    """Save the analysis results to a CSV file."""
    fieldnames = [
        "Language",
        "Sample Size",
        "Token Count",
        "Text Type",
        "Initial Gini",
        "Final Gini",
        "Initial Entropy",
        "Final Entropy",
        "Gini Difference",
        "Entropy Difference",
        "Combined Score",  # Added this field
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
    """Main function to orchestrate the analysis of writing directionality."""
    download_nltk_corpora()
    
    available_languages = get_available_languages()
    if not available_languages:
        print("No languages found in 'europarl_raw' corpus.")
        sys.exit(1)
    
    print(f"Available languages in 'europarl_raw': {available_languages}\n")
    
    all_results = process_languages(available_languages)
    
    display_results(all_results)
    
    # Ask if the user wants to save the results to CSV
    save_choice = input("\nWould you like to save the results to 'directionality_results.csv'? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_results_to_csv(all_results)
    else:
        print("Results not saved.")


if __name__ == "__main__":
    main()
