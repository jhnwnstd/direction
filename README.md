# Directionality Analysis of Writing

This repository contains a Python script for analyzing the directionality of writing by examining the character frequency and entropy differences in the initial and final positions of words. This analysis is based on research from the paper, *The "Handedness" of Language: Directional Symmetry Breaking of Sign Usage in Words*, by Md Izhar Ashraf and Sitabhra Sinha. Using techniques from the paper, this script processes texts from multiple corpora to determine if a language is predominantly left-to-right or right-to-left.

## Files in this Repository

- **direction.py**: The main Python script that performs the directionality analysis.
- **The_handedness_of_language_Directional.pdf**: The foundational research paper, detailing the theory and empirical analysis of directional symmetry in languages.
- **results.csv**: Comprehensive analysis results across multiple languages.

## Project Background

The script is inspired by Ashraf and Sinha's work, which identified a universal asymmetry in character distributions across languages. Their research demonstrates that:

- **Initial characters** of words have a more balanced distribution, indicating greater flexibility in sign choice.
- **Final characters** show a more restrictive distribution, meaning fewer characters typically occupy this position.
  
This directional asymmetry—measured using statistical metrics such as the **Gini coefficient** and **entropy**—provides insights into whether a language is written left-to-right or right-to-left. By analyzing these metrics, we can predict the writing direction of a language with high accuracy.

## Methodology

The directionality analysis employs the following steps:

1. **Data Collection**:
   - Utilizes the `europarl_raw` and `udhr` corpora from NLTK to gather text samples from various languages.
   - Extracts a fixed sample size (default 750 characters) from each corpus for analysis.

2. **Character Frequency Extraction**:
   - **Initial Characters**: Extracts the first character of each word to analyze their distribution.
   - **Final Characters**: Extracts the last character of each word to analyze their distribution.

3. **Statistical Metrics Calculation**:
   - **Gini Coefficient**: Measures the inequality in character distributions. A lower Gini coefficient indicates a more balanced distribution.
   - **Entropy**: Measures the randomness or unpredictability in character distributions. Higher entropy signifies greater randomness.

4. **Directionality Prediction**:
   - **Left-to-Right (LTR)**: Identified when the **Gini Difference** (Initial Gini - Final Gini) is negative and the **Entropy Difference** (Initial Entropy - Final Entropy) is positive.
   - **Right-to-Left (RTL)**: Identified when the **Gini Difference** is positive and the **Entropy Difference** is negative.
   - **Indeterminate**: When the differences do not clearly indicate a directionality.

5. **Reversed Text Analysis**:
   - To validate the methodology, the script also analyzes reversed text samples. This ensures that the directional metrics flip accordingly, confirming the robustness of the analysis.

6. **Result Compilation**:
   - Aggregates the analysis results into a CSV file for comprehensive review and further research.

## Key Findings

Analysis of 13 languages revealed consistent patterns that validate the methodology. Here are the complete results:

| Language   | Sample Size | Text Type | Initial Gini | Final Gini | Initial Entropy | Final Entropy | Gini Difference | Entropy Difference | Likely Direction |
|------------|-------------|------------|--------------|------------|-----------------|---------------|-----------------|-------------------|-----------------|
| Danish     | 750         | Normal     | 0.4391       | 0.4843     | 4.1322         | 3.3451        | -0.0452         | 0.7871            | Left-to-Right   |
| Danish     | 750         | Reversed   | 0.4843       | 0.4391     | 3.3451         | 4.1322        | 0.0452          | -0.7871           | Right-to-Left   |
| Dutch      | 750         | Normal     | 0.4505       | 0.5055     | 3.9709         | 3.0675        | -0.0550         | 0.9034            | Left-to-Right   |
| Dutch      | 750         | Reversed   | 0.5055       | 0.4505     | 3.0675         | 3.9709        | 0.0550          | -0.9034           | Right-to-Left   |
| English    | 750         | Normal     | 0.3716       | 0.5193     | 3.9666         | 3.4086        | -0.1477         | 0.5581            | Left-to-Right   |
| English    | 750         | Reversed   | 0.5193       | 0.3716     | 3.4086         | 3.9666        | 0.1477          | -0.5581           | Right-to-Left   |
| Finnish    | 750         | Normal     | 0.3359       | 0.4985     | 3.8126         | 2.3642        | -0.1626         | 1.4484            | Left-to-Right   |
| Finnish    | 750         | Reversed   | 0.4985       | 0.3359     | 2.3642         | 3.8126        | 0.1626          | -1.4484           | Right-to-Left   |
| French     | 750         | Normal     | 0.4012       | 0.6053     | 4.0665         | 2.9424        | -0.2041         | 1.1241            | Left-to-Right   |
| French     | 750         | Reversed   | 0.6053       | 0.4012     | 2.9424         | 4.0665        | 0.2041          | -1.1241           | Right-to-Left   |
| German     | 750         | Normal     | 0.4223       | 0.5609     | 4.0246         | 2.9083        | -0.1385         | 1.1163            | Left-to-Right   |
| German     | 750         | Reversed   | 0.5609       | 0.4223     | 2.9083         | 4.0246        | 0.1385          | -1.1163           | Right-to-Left   |
| Greek      | 750         | Normal     | 0.4911       | 0.5529     | 4.0549         | 3.0230        | -0.0618         | 1.0319            | Left-to-Right   |
| Greek      | 750         | Reversed   | 0.5529       | 0.4911     | 3.0230         | 4.0549        | 0.0618          | -1.0319           | Right-to-Left   |
| Italian    | 750         | Normal     | 0.4742       | 0.5219     | 3.7092         | 2.4568        | -0.0477         | 1.2524            | Left-to-Right   |
| Italian    | 750         | Reversed   | 0.5219       | 0.4742     | 2.4568         | 3.7092        | 0.0477          | -1.2524           | Right-to-Left   |
| Portuguese | 750         | Normal     | 0.4183       | 0.5974     | 3.9604         | 2.6653        | -0.1790         | 1.2951            | Left-to-Right   |
| Portuguese | 750         | Reversed   | 0.5974       | 0.4183     | 2.6653         | 3.9604        | 0.1790          | -1.2951           | Right-to-Left   |
| Spanish    | 750         | Normal     | 0.4682       | 0.5281     | 3.7903         | 2.6029        | -0.0599         | 1.1874            | Left-to-Right   |
| Spanish    | 750         | Reversed   | 0.5281       | 0.4682     | 2.6029         | 3.7903        | 0.0599          | -1.1874           | Right-to-Left   |
| Swedish    | 750         | Normal     | 0.4063       | 0.5230     | 4.2511         | 3.3237        | -0.1167         | 0.9275            | Left-to-Right   |
| Swedish    | 750         | Reversed   | 0.5230       | 0.4063     | 3.3237         | 4.2511        | 0.1167          | -0.9275           | Right-to-Left   |
| Arabic     | 750         | Normal     | 0.5121       | 0.4185     | 3.5498         | 3.8864        | 0.0935          | -0.3366           | Right-to-Left   |
| Arabic     | 750         | Reversed   | 0.4185       | 0.5121     | 3.8864         | 3.5498        | -0.0935         | 0.3366            | Left-to-Right   |
| Hebrew     | 750         | Normal     | 0.5265       | 0.4938     | 3.3217         | 3.5849        | 0.0327          | -0.2633           | Right-to-Left   |
| Hebrew     | 750         | Reversed   | 0.4938       | 0.5265     | 3.5849         | 3.3217        | -0.0327         | 0.2633            | Left-to-Right   |

### Explanation of Results

1. **Left-to-Right (LTR) Languages**:
   - **Gini Difference**: Negative values indicate that the initial characters have a more balanced distribution compared to the final characters.
   - **Entropy Difference**: Positive values suggest higher randomness in initial characters and lower randomness in final characters.
   - **Interpretation**: The combination of these differences aligns with the expected patterns for LTR languages, where the start of words allows for more variability, and the endings are more constrained.

2. **Right-to-Left (RTL) Languages**:
   - **Gini Difference**: Positive values signify that the initial characters (which are actually the word endings in RTL scripts) have a more restricted distribution.
   - **Entropy Difference**: Negative values indicate lower randomness in the initial (end) characters and higher randomness in the final (start) characters.
   - **Interpretation**: These metrics are the inverse of LTR languages, accurately reflecting the RTL writing direction.

3. **Reversed Text Analysis**:
   - Consistently showed flipped Gini and entropy differences, validating that the metrics are direction-sensitive.
   - **Direction Classification**: Reversed texts correctly classified the opposite direction, reinforcing the reliability of the methodology.

### Implications for Decipherment

The findings from this analysis hold significant implications for the field of **decipherment**, particularly in the study of ancient or unknown scripts. Here's how:

1. **Determining Writing Direction**:
   - **Challenge**: One of the primary challenges in deciphering unknown scripts is identifying the writing direction.
   - **Solution**: By analyzing character frequency and entropy at word boundaries, researchers can infer the likely writing direction even without knowing the language.

2. **Guiding Decipherment Efforts**:
   - **Hypothesis Testing**: Establishing the directionality can help in forming hypotheses about the script's structure, such as identifying potential word boundaries and parsing text correctly.
   - **Pattern Recognition**: Understanding whether a script is LTR or RTL can aid in recognizing repeating patterns, letter frequencies, and potential linguistic similarities with known languages.

3. **Enhancing Computational Tools**:
   - **Automated Analysis**: Scripts like `direction.py` can be integrated into larger computational frameworks to provide preliminary analyses, saving time and resources in the decipherment process.
   - **Cross-Linguistic Insights**: Comparing directional asymmetries across languages can uncover universal patterns that might apply to unknown scripts, offering a starting point for deeper analysis.

4. **Limitations and Future Work**:
   - **Script Variations**: While the methodology is robust for determining horizontal writing directions, it may need adaptations for vertical or boustrophedon scripts.
   - **Complex Scripts**: Languages with mixed or flexible writing directions might present challenges, necessitating more nuanced analysis techniques.

In summary, the directionality analysis provides a valuable tool for linguists and cryptographers in deciphering unknown writing systems by offering a systematic approach to infer writing directions based on statistical character distributions.

## Features

- **Automatic Language Detection**: Automatically identifies and processes available languages from various corpora.
- **Gini Coefficient and Entropy Calculation**: Computes statistical measures to analyze the inequality and randomness of character distributions at word boundaries.
- **Directionality Prediction**: Classifies each language as left-to-right, right-to-left, or indeterminate based on Gini and entropy differences.
- **CSV Output**: Saves comprehensive analysis results to a CSV file for further study.

## Requirements

- **Python 3.7+**
- **Required Packages**:
  - `nltk`
  - `numpy`
  - `scipy`

## Usage

1. **Install Required Packages**:
   ```bash
   pip install nltk numpy scipy
   ```

2. **Run the Script**:
   ```bash
   python direction.py
   ```

3. **View Results**:
   Results will be displayed in the console and saved to a CSV file containing:
   - Sample size and text type
   - Initial and final Gini coefficients
   - Initial and final entropy values
   - Gini and entropy differences
   - Predicted writing direction

## Research Basis

This analysis is rooted in the study by Ashraf and Sinha, which uses the Gini coefficient and entropy to reveal a universal "handedness" in word structure across languages and writing systems. Our results strongly validate their methodology, showing clear directional signals across both left-to-right and right-to-left writing systems.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. Areas for potential enhancement include:

- Adding support for additional corpora
- Implementing analysis of vertical writing systems
- Developing visualization tools for results
- Expanding language coverage

## License

This project is open-source and available under the MIT License.

## Author

Created by [John Winstead](https://github.com/jhnwnstd).