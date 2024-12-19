# Standard Library Imports
import sys
import csv
from collections import Counter
from typing import List, Tuple, Dict

# Third-Party Imports
import nltk
from nltk.corpus import europarl_raw, udhr
import numpy as np
from scipy.stats import entropy


def download_nltk_corpora() -> None:
    """
    Ensure that the necessary NLTK corpora are downloaded.
    
    This function checks for the presence of the 'europarl_raw' and 'udhr' corpora.
    If any of them are missing, it downloads them using NLTK's downloader.
    """
    # List of required NLTK corpora
    required_corpora = ['europarl_raw', 'udhr']
    
    for corpus_name in required_corpora:
        try:
            # Attempt to find the corpus in NLTK's data directory
            nltk.data.find(f'corpora/{corpus_name}')
        except LookupError:
            # If not found, download the missing corpus
            print(f"Downloading '{corpus_name}' corpus...")
            nltk.download(corpus_name)


def get_available_languages() -> List[str]:
    """
    Retrieve a list of available languages in the europarl_raw corpus.
    
    This function inspects the europarl_raw module to identify all available language corpora.
    
    Returns:
        List[str]: A list of language names available in europarl_raw.
    """
    # List comprehension to gather language names that have a 'raw' method
    languages = [
        lang for lang in dir(europarl_raw)
        if lang.islower() and hasattr(getattr(europarl_raw, lang), 'raw')
    ]
    return languages


def calculate_gini_coefficient(freqs: List[int]) -> float:
    """
    Calculate the Gini coefficient for a list of frequencies.
    
    The Gini coefficient measures the inequality among values of a frequency distribution.
    A higher Gini coefficient indicates greater inequality.
    
    Args:
        freqs (List[int]): A list of frequency counts.
    
    Returns:
        float: The calculated Gini coefficient.
    """
    # Sort the frequencies in ascending order
    sorted_freqs = np.sort(freqs)
    n = len(sorted_freqs)
    
    if n == 0:
        # If there are no frequencies, return 0.0 as the Gini coefficient
        return 0.0
    
    # Compute the cumulative sum of frequencies
    cumulative_sum = np.cumsum(sorted_freqs)
    total = cumulative_sum[-1]
    
    if total == 0:
        # If the total frequency is zero, return 0.0
        return 0.0
    
    # Create an array of indices starting from 1 to n
    index = np.arange(1, n + 1)
    
    # Calculate the Gini coefficient using the formula
    gini = (2 * np.sum(index * sorted_freqs)) / (n * total) - (n + 1) / n
    return gini


def calculate_entropy_value(freqs: List[int]) -> float:
    """
    Calculate entropy based on the frequency of elements.
    
    Entropy measures the randomness or unpredictability in the distribution of characters.
    
    Args:
        freqs (List[int]): A list of frequency counts.
    
    Returns:
        float: The calculated entropy value.
    """
    if not freqs:
        # If there are no frequencies, entropy is zero
        return 0.0
    
    # Calculate entropy using base 2 logarithm
    return entropy(freqs, base=2)


def extract_character_frequencies(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract initial and final character frequencies from the provided text.
    
    This function processes the text to extract the first and last characters of each word,
    excluding single-character words as they do not provide meaningful directional information.
    
    Args:
        text (str): The input text to analyze.
    
    Returns:
        Tuple[List[str], List[str]]: Two lists containing initial and final characters respectively.
    """
    # Split the text into words and remove any surrounding whitespace
    words = [word.strip() for word in text.split() if word]
    
    # Extract the first character of each word with more than one character, converted to lowercase
    initial_chars = [word[0].lower() for word in words if len(word) > 1]
    
    # Extract the last character of each word with more than one character, converted to lowercase
    final_chars = [word[-1].lower() for word in words if len(word) > 1]
    
    return initial_chars, final_chars


def analyze_directionality(text: str) -> Dict[str, float]:
    """
    Analyze the text to predict the direction of the writing system.
    
    This function computes various statistical measures, including entropy and Gini coefficients,
    to determine whether the text is written from Left-to-Right (LTR) or Right-to-Left (RTL).
    
    Args:
        text (str): The input text to analyze.
    
    Returns:
        Dict[str, float]: A dictionary containing calculated metrics and the likely writing direction.
    """
    # Extract initial and final characters from the text
    initial_chars, final_chars = extract_character_frequencies(text)
    
    if not initial_chars or not final_chars:
        # Raise an error if no valid words are found for analysis
        raise ValueError("No valid words found in the text for analysis.")
    
    # Count the frequency of each initial character
    initial_freqs = Counter(initial_chars)
    
    # Count the frequency of each final character
    final_freqs = Counter(final_chars)
    
    # Calculate Gini coefficients for initial and final character distributions
    initial_gini = calculate_gini_coefficient(list(initial_freqs.values()))
    final_gini = calculate_gini_coefficient(list(final_freqs.values()))
    
    # Calculate entropy values for initial and final character distributions
    initial_entropy = calculate_entropy_value(list(initial_freqs.values()))
    final_entropy = calculate_entropy_value(list(final_freqs.values()))
    
    # Compute differences between initial and final metrics
    gini_difference = initial_gini - final_gini
    entropy_difference = initial_entropy - final_entropy
    
    # Calculate the maximum possible entropy based on the number of unique characters
    max_entropy = np.log2(len(set(initial_chars + final_chars))) if (initial_chars + final_chars) else 0.0
    
    # Normalize Gini difference to a [0, 1] scale
    normalized_gini_diff = (gini_difference + 1) / 2  # Assuming Gini coefficient ranges from -1 to 1
    
    # Normalize entropy difference to a [0, 1] scale based on maximum possible entropy
    normalized_entropy_diff = (
        (entropy_difference + max_entropy) / (2 * max_entropy) if max_entropy else 0.0
    )
    
    # Combine normalized entropy and Gini differences to compute the final score
    combined_score = normalized_entropy_diff - normalized_gini_diff
    
    # Determine the likely writing direction based on the combined score
    if combined_score > 0:
        likely_direction = "Left-to-Right"
    elif combined_score < 0:
        likely_direction = "Right-to-Left"
    else:
        likely_direction = "Indeterminate"
    
    # Compile all calculated metrics into a dictionary
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


def process_languages(
    languages: List[str],
    europarl_sample_size: int = None,
    udhr_sample_size: int = None
) -> List[Dict[str, float]]:
    """
    Process each language in Europarl and UDHR corpora, including reversed text testing.
    
    This function analyzes both normal and reversed versions of each language's text to validate
    the directionality analysis.
    
    Args:
        languages (List[str]): A list of language names to process.
        europarl_sample_size (int, optional): Number of characters to sample from Europarl corpus. Defaults to None.
        udhr_sample_size (int, optional): Number of characters to sample from UDHR corpus. Defaults to None.
    
    Returns:
        List[Dict[str, float]]: A list of dictionaries containing analysis results for each language.
    """
    results = []  # Initialize a list to store analysis results
    
    # Iterate over each language in the Europarl corpus
    for language in languages:
        print(f"Processing Language: {language.capitalize()}")
        try:
            # Access the raw text data for the current language
            corpus = getattr(europarl_raw, language)
            text_data = corpus.raw()
            token_count = len(text_data.split())  # Count the number of tokens (words)
    
            if europarl_sample_size is not None:
                # If a sample size is specified, truncate the text accordingly
                text_data = text_data[:europarl_sample_size]
    
            # Create a reversed version of the text for validation
            reversed_text_data = text_data[::-1]
    
            # Analyze the normal text directionality
            normal_results = analyze_directionality(text_data)
            normal_results.update({
                "Language": language.capitalize(),
                "Sample Size": len(text_data),
                "Text Type": "Normal",
                "Token Count": token_count
            })
            results.append(normal_results)
    
            # Analyze the reversed text directionality
            reversed_results = analyze_directionality(reversed_text_data)
            reversed_results.update({
                "Language": language.capitalize(),
                "Sample Size": len(reversed_text_data),
                "Text Type": "Reversed",
                "Token Count": token_count
            })
            results.append(reversed_results)
    
            print(f"Analysis for {language.capitalize()} completed.\n")
    
        except Exception as e:
            # Handle any errors that occur during processing
            print(f"Error processing {language}: {e}\n")
    
    # List of additional languages from the UDHR corpus to process
    udhr_languages = [
        ('Arabic', 'Arabic_Alarabia-Arabic'),
        ('Hebrew', 'Hebrew_Ivrit-Hebrew')
    ]
    
    # Iterate over each UDHR language
    for udhr_language, fileid in udhr_languages:
        print(f"Processing UDHR Language: {udhr_language}")
        try:
            # Access the raw text data for the UDHR language
            text_data = udhr.raw(fileids=fileid)
            token_count = len(text_data.split())  # Count the number of tokens (words)
    
            if udhr_sample_size is not None:
                # If a sample size is specified, truncate the text accordingly
                text_data = text_data[:udhr_sample_size]
    
            # Create a reversed version of the text for validation
            reversed_text_data = text_data[::-1]
    
            # Analyze the normal text directionality
            normal_results = analyze_directionality(text_data)
            normal_results.update({
                "Language": udhr_language,
                "Sample Size": len(text_data),
                "Text Type": "Normal",
                "Token Count": token_count
            })
            results.append(normal_results)
    
            # Analyze the reversed text directionality
            reversed_results = analyze_directionality(reversed_text_data)
            reversed_results.update({
                "Language": udhr_language,
                "Sample Size": len(reversed_text_data),
                "Text Type": "Reversed",
                "Token Count": token_count
            })
            results.append(reversed_results)
    
            print(f"Analysis for {udhr_language} completed.\n")
    
        except Exception as e:
            # Handle any errors that occur during processing
            print(f"Error processing {udhr_language}: {e}\n")
    
    return results


def display_results(results: List[Dict[str, float]]) -> None:
    """
    Display the analysis results in a tabular format.
    
    This function prints the results to the standard output in a tab-separated format.
    
    Args:
        results (List[Dict[str, float]]): The list of analysis result dictionaries to display.
    """
    if not results:
        # Inform the user if there are no results to display
        print("No results to display.")
        return
    
    print("\n--- Comprehensive Analysis Results ---\n")
    
    # Define the order and names of the fields for the CSV output
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
        "Normalized Gini Difference",
        "Normalized Entropy Difference",
        "Combined Score",
        "Likely Direction"
    ]
    
    # Initialize a CSV DictWriter to output the results to stdout with tab delimiter
    csv_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter='\t')
    csv_writer.writeheader()  # Write the header row
    
    # Iterate over each result and write it as a row in the CSV
    for result in results:
        csv_writer.writerow(result)


def save_results_to_csv(results: List[Dict[str, float]], filename: str = 'directionality_results.csv') -> None:
    """
    Save the analysis results to a CSV file.
    
    This function writes the results to a specified CSV file with UTF-8 encoding.
    
    Args:
        results (List[Dict[str, float]]): The list of analysis result dictionaries to save.
        filename (str, optional): The name of the output CSV file. Defaults to 'directionality_results.csv'.
    """
    # Define the order and names of the fields for the CSV output
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
        "Normalized Gini Difference",
        "Normalized Entropy Difference",
        "Combined Score",
        "Likely Direction"
    ]
    
    try:
        # Open the specified CSV file in write mode with UTF-8 encoding
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Initialize a CSV DictWriter with the defined fieldnames
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  # Write the header row
            
            # Iterate over each result and write it as a row in the CSV
            for result in results:
                writer.writerow(result)
        
        print(f"Results successfully saved to '{filename}'.")
    
    except Exception as e:
        # Handle any errors that occur during the file writing process
        print(f"Error saving to CSV: {e}")


def main() -> None:
    """
    Main function to orchestrate the directionality analysis.
    
    This function coordinates the workflow by downloading necessary corpora, retrieving
    available languages, processing each language's text, displaying the results, and
    optionally saving them to a CSV file based on user input.
    """
    # Ensure that the required NLTK corpora are available
    download_nltk_corpora()
    
    # Retrieve the list of available languages from the Europarl corpus
    languages = get_available_languages()
    
    if not languages:
        # Exit the program if no languages are found in the Europarl corpus
        print("No languages found in 'europarl_raw' corpus.")
        sys.exit(1)
    
    # Display the list of available languages to the user
    print(f"Available languages: {languages}\n")
    
    # Process each language and collect the analysis results
    all_results = process_languages(languages)
    
    # Display the collected results in a tabular format
    display_results(all_results)
    
    # Prompt the user to decide whether to save the results to a CSV file
    save_choice = input("\nWould you like to save the results to 'directionality_results.csv'? (y/n): ").strip().lower()
    
    if save_choice == 'y':
        # If the user chooses to save, call the function to save results
        save_results_to_csv(all_results)
    else:
        # Inform the user that results are not saved
        print("Results not saved.")


if __name__ == "__main__":
    main()