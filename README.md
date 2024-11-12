# Directionality Analysis of Languages

This repository contains a Python script for analyzing the directionality of languages by examining the character frequency and entropy differences in the initial and final positions of words. This analysis is based on research from the paper, *The “Handedness” of Language: Directional Symmetry Breaking of Sign Usage in Words*, by Md Izhar Ashraf and Sitabhra Sinha. Using techniques from the paper, this script processes texts from the Europarl corpus to determine if a language is predominantly left-to-right or right-to-left.

## Files in this Repository

- **direction.py**: The main Python script that performs the directionality analysis.
- **The_handedness_of_language_Directional.pdf**: The foundational research paper, detailing the theory and empirical analysis of directional symmetry in languages.

## Project Background

The script is inspired by Ashraf and Sinha’s work, which identified a universal asymmetry in character distributions across languages. Their research demonstrates that:
- **Initial characters** of words have a more balanced distribution, indicating greater flexibility in sign choice.
- **Final characters** show a more restrictive distribution, meaning fewer characters typically occupy this position.
  
This directional asymmetry—measured using statistical metrics such as the **Gini coefficient** and **entropy**—provides insights into whether a language is written left-to-right or right-to-left.

## Features

- **Automatic Language Detection**: Automatically identifies and processes available languages from the Europarl corpus.
- **Gini Coefficient and Entropy Calculation**: Computes statistical measures to analyze the inequality and randomness of character distributions at word boundaries.
- **Directionality Prediction**: Classifies each language as left-to-right, right-to-left, or indeterminate based on Gini and entropy differences.
- **CSV Output**: Optionally saves the analysis results to a CSV file for further study.

## Requirements

- **Python 3.7+**
- **Required Packages**:
  - `nltk`
  - `numpy`
  - `scipy`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. **Download the NLTK Europarl Corpus**:
   The script will automatically download the Europarl corpus if it’s not already present on your system.

2. **Run the Script**:
   ```bash
   python direction.py
   ```

3. **Follow the On-Screen Prompts**:
   The script will display a list of available languages in the Europarl corpus and process each language, displaying the directionality results in a tabular format. You will also have the option to save the results to a CSV file.

## Example Output

```
--- Comprehensive Analysis Results ---

Language        Initial Gini    Final Gini      Initial Entropy Final Entropy   Gini Difference Entropy Difference      Likely Direction
Danish          0.6033          0.7836          4.2522         3.5004          -0.1803        0.7518                  Left-to-Right
...
```

## Research Basis

This analysis is rooted in the study by Ashraf and Sinha, which uses the Gini coefficient and entropy to reveal a universal "handedness" in word structure across languages and writing systems. The approach presented here replicates their methodology for use with the Europarl corpus, providing a practical application of their findings.

The findings in *The “Handedness” of Language* support the notion that the beginning of words tends to have more variety in character usage, while the end of words is more constrained, reflecting an inherent cognitive pattern in human language processing.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. Feel free to report issues or suggest enhancements!

## License

This project is open-source and available under the MIT License.

## Author

Created by [John Winstead](https://github.com/jhnwnstd).