import nltk
from nltk.corpus import europarl_raw
from collections import Counter
import numpy as np
from scipy.stats import entropy
import sys
import csv

def download_nltk_corpora():
    """
    Download necessary NLTK corpora if not already present.
    """
    try:
        nltk.data.find('corpora/europarl_raw')
    except LookupError:
        print("Downloading 'europarl_raw' corpus...")
        nltk.download('europarl_raw')

def get_available_languages():
    """
    Retrieve a list of available languages in the europarl_raw corpus.
    Only include languages that have a callable 'raw' method.
    """
    languages = []
    for lang in dir(europarl_raw):
        if lang.islower():
            attr = getattr(europarl_raw, lang, None)
            if hasattr(attr, 'raw') and callable(attr.raw):
                languages.append(lang)
    return languages

def gini_coefficient(freqs):
    """Calculate the Gini coefficient for a given list of frequencies."""
    sorted_freqs = np.sort(freqs)  # Sort in ascending order
    n = len(sorted_freqs)
    if n == 0:
        return 0.0
    cumulative_sum = np.cumsum(sorted_freqs)
    sum_x = cumulative_sum[-1]
    if sum_x == 0:
        return 0.0
    # Gini formula
    G = (np.sum((2 * np.arange(1, n + 1) * sorted_freqs))) / (n * sum_x) - (n + 1) / n
    return G

def calculate_entropy(freqs):
    """Calculate entropy based on the frequency of elements."""
    if not freqs:
        return 0.0
    return entropy(freqs, base=2)

def process_text(text):
    """Process text to extract initial and final character frequencies."""
    words = [word.strip() for word in text.split() if word]
    initial_chars = [word[0].lower() for word in words if len(word) > 1]
    final_chars = [word[-1].lower() for word in words if len(word) > 1]
    return initial_chars, final_chars

def analyze_direction(text):
    """Analyze text to predict the direction of the writing system."""
    # Step 1: Extract initial and final characters from the words
    initial_chars, final_chars = process_text(text)
    
    if not initial_chars or not final_chars:
        print("No valid words found in the text for analysis.")
        sys.exit(1)
    
    # Step 2: Calculate frequencies for initial and final characters
    initial_freqs = Counter(initial_chars)
    final_freqs = Counter(final_chars)
    
    # Ensure there are frequencies to analyze
    if not initial_freqs or not final_freqs:
        print("Insufficient character frequency data for analysis.")
        sys.exit(1)
    
    # Step 3: Calculate Gini coefficient and entropy for initial and final characters
    initial_gini = gini_coefficient(list(initial_freqs.values()))
    final_gini = gini_coefficient(list(final_freqs.values()))
    
    initial_entropy = calculate_entropy(list(initial_freqs.values()))
    final_entropy = calculate_entropy(list(final_freqs.values()))
    
    # Step 4: Infer writing direction based on Gini and entropy differences
    gini_difference = initial_gini - final_gini
    entropy_difference = initial_entropy - final_entropy
    
    # Step 5: Determine likely direction based on Gini and entropy values
    if gini_difference < 0 and entropy_difference > 0:
        likely_direction = "Left-to-Right"
    elif gini_difference > 0 and entropy_difference < 0:
        likely_direction = "Right-to-Left"
    else:
        likely_direction = "Indeterminate"

    # Results summary
    results = {
        "Initial Gini": round(initial_gini, 4),
        "Final Gini": round(final_gini, 4),
        "Initial Entropy": round(initial_entropy, 4),
        "Final Entropy": round(final_entropy, 4),
        "Gini Difference": round(gini_difference, 4),
        "Entropy Difference": round(entropy_difference, 4),
        "Likely Direction": likely_direction
    }

    return results

def main():
    # Step 0: Ensure necessary NLTK corpora are downloaded
    download_nltk_corpora()
    
    # Step 1: Retrieve all available languages
    available_languages = get_available_languages()
    if not available_languages:
        print("No languages found in 'europarl_raw' corpus.")
        sys.exit(1)
    
    print(f"Available languages in 'europarl_raw': {available_languages}\n")
    
    # Prepare to collect results
    all_results = []
    
    # Step 2: Iterate through each language and perform analysis
    for language in available_languages:
        print(f"Processing Language: {language.capitalize()}")
        try:
            # Load a sample from the Europarl corpus
            text_data = getattr(europarl_raw, language).raw()[:1000]  # Adjust sample size as needed
            print("Corpus loaded successfully.")
            
            # Analyze directionality
            results = analyze_direction(text_data)
            
            # Add language info to results
            results["Language"] = language.capitalize()
            all_results.append(results)
            
            print("Analysis completed successfully.\n")
        
        except AttributeError:
            print(f"Language '{language}' not found in 'europarl_raw'. Skipping.\n")
            continue
        except ValueError as ve:
            print(f"Analysis Error: {ve}. Skipping.\n")
            continue
        except Exception as e:
            print(f"Unexpected Error: {e}. Skipping.\n")
            continue
    
    # Step 3: Display all results in a table format
    if all_results:
        print("\n--- Comprehensive Analysis Results ---\n")
        # Define the order of columns
        fieldnames = ["Language", "Initial Gini", "Final Gini", "Initial Entropy", 
                      "Final Entropy", "Gini Difference", "Entropy Difference", "Likely Direction"]
        
        # Print the results in a tabular format using csv module
        csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter='\t')
        csv_writer.writeheader()
        for result in all_results:
            csv_writer.writerow(result)
        
        # Optionally, save the results to a CSV file
        save_to_csv = input("\nWould you like to save the results to 'directionality_results.csv'? (y/n): ").strip().lower()
        if save_to_csv == 'y':
            try:
                with open('directionality_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in all_results:
                        writer.writerow(result)
                print("Results successfully saved to 'directionality_results.csv'.")
            except Exception as e:
                print(f"Error saving to CSV: {e}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
