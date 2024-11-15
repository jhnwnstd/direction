# Directionality Analysis of Writing

This repository contains Python tools for analyzing writing directionality by examining character frequency and entropy differences in word-initial and word-final positions. Based on research from *The "Handedness" of Language: Directional Symmetry Breaking of Sign Usage in Words* by Md Izhar Ashraf and Sitabhra Sinha, it can automatically detect whether a language is Left-to-Right (LTR) or Right-to-Left (RTL).

## Files in Repository

- **direction.py**: Main Python script performing directionality analysis
- **The_handedness_of_language_Directional.pdf**: Original research paper detailing theory and empirical analysis
- **results.csv**: Comprehensive analysis results across multiple languages

## Project Background

The analysis is rooted in Ashraf and Sinha's discovery of universal asymmetry in character distributions across languages:
- **Initial characters** show more balanced distribution (greater flexibility)
- **Final characters** show more restrictive distribution (fewer common endings)

This directional asymmetry—measured using statistical metrics—provides insights into writing direction, with implications for decipherment of unknown scripts.

## Methodology

### Data Collection
1. Utilizes `europarl_raw` and `udhr` corpora from NLTK
2. Extracts text samples:
   - European languages (Europarl): 100,000 characters
   - RTL languages (UDHR): Full text available

### Character Analysis
1. **Initial Characters**: Extracts first character of each word
2. **Final Characters**: Extracts last character of each word
3. Analyzes distribution patterns using statistical measures

### Statistical Metrics
For each text sample, calculates:
- **Entropy**: Measures randomness/predictability of character distributions
- **Gini Coefficient**: Measures inequality in character frequency distributions
- **Combined Score**: Weighted combination indicating directionality

### Scoring System
The improved scoring system combines:
1. **Entropy Difference** = Initial Entropy - Final Entropy
2. **Scaled Gini Difference** = (Initial Gini - Final Gini) × 5.0
3. **Combined Score** = Entropy Difference - Scaled Gini Difference

Score interpretation:
- Positive → Left-to-Right writing system
- Negative → Right-to-Left writing system
- Magnitude indicates strength of directional signal

## Results

### European Languages (LTR)
All correctly identified as Left-to-Right:
- **Strongest signals**:
  - Finnish (1.456)
  - Italian (1.3576)
  - Portuguese (1.3555)
- **Moderate signals**:
  - French (1.0639)
  - Dutch (1.1662)
  - German (0.9778)
  - Danish (0.932)
  - Swedish (0.9255)
  - Spanish (1.068)
- **Weakest signal**:
  - Greek (0.5617)

### Semitic Languages (RTL)
Both correctly identified as Right-to-Left:
- **Arabic**: Strong RTL signal (-0.502)
- **Hebrew**: Weaker RTL signal (-0.0664)

### Validation
- Reversed text samples show opposite directionality
- Score magnitudes preserved in reversed text
- Pattern consistent across all languages tested
- European languages show stronger directional signals than Semitic languages

## Implications for Decipherment

### Challenges Addressed
1. **Writing Direction Determination**
   - Primary challenge in unknown script analysis
   - Solution: Statistical analysis of character distributions

2. **Pattern Recognition**
   - Aids in identifying word boundaries
   - Helps recognize linguistic patterns

3. **Computational Applications**
   - Automated preliminary analysis
   - Integration with larger decipherment frameworks

### Limitations
- May need adaptation for vertical scripts
- Challenges with mixed/flexible writing directions
- Requires sufficient sample size

## Implementation

```python
def analyze_directionality(text):
    """Analyze text directionality using entropy and Gini coefficient."""
    # Extract character frequencies
    initial_chars, final_chars = extract_character_frequencies(text)
    
    # Calculate statistical measures
    initial_gini = calculate_gini_coefficient(initial_chars)
    final_gini = calculate_gini_coefficient(final_chars)
    initial_entropy = calculate_entropy_value(initial_chars)
    final_entropy = calculate_entropy_value(final_chars)
    
    # Calculate differences with Gini scaling
    gini_difference = (initial_gini - final_gini) * 5.0
    entropy_difference = initial_entropy - final_entropy
    combined_score = entropy_difference - gini_difference
    
    return {
        "Initial Gini": initial_gini,
        "Final Gini": final_gini,
        "Initial Entropy": initial_entropy,
        "Final Entropy": final_entropy,
        "Gini Difference": gini_difference,
        "Entropy Difference": entropy_difference,
        "Combined Score": combined_score,
        "Likely Direction": "Left-to-Right" if combined_score > 0 else "Right-to-Left"
    }
```

## Features

- Automatic language detection
- Statistical analysis of character distributions
- Directionality prediction
- Comprehensive CSV output
- Validation through reversed text analysis

## Dependencies

Required packages:
- NLTK (corpora access)
- NumPy (statistical calculations)
- SciPy (entropy calculations)

## Installation

```bash
pip install nltk numpy scipy
```

## Usage

```python
from direction import analyze_directionality

# Analyze a text sample
results = analyze_directionality(text_sample)

# Results include:
# - Initial/Final Gini coefficients
# - Initial/Final Entropy values
# - Gini Difference (scaled)
# - Entropy Difference
# - Combined Score
# - Likely Direction
```

## Limitations

1. **Technical Constraints**:
   - Requires sufficient text sample size
   - Performance varies with text genre/formality
   - Some languages show weaker directional signals
   - Sample size differences between corpora affect comparability

2. **Methodological Limitations**:
   - Not designed for vertical writing systems
   - May struggle with mixed-direction scripts
   - Requires clean, well-formatted text input

## Future Improvements

### Technical Enhancements
1. Data Processing:
   - Normalize for corpus size differences
   - Optimize Gini coefficient scaling
   - Add confidence scores for predictions

2. Feature Additions:
   - Support for vertical writing systems
   - Additional statistical measures
   - Interactive visualization tools
   - Extended language coverage

### Research Extensions
1. Script Analysis:
   - Support for additional writing systems
   - Historical script analysis tools
   - Comparative analysis features

2. Tool Development:
   - GUI for analysis visualization
   - Batch processing capabilities
   - Integration with other linguistic tools

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Submit pull request

Areas for contribution:
- Additional corpora support
- Vertical writing system analysis
- Visualization tools
- Language coverage expansion

## License

This project is open-source under the MIT License.

## Author

Created by [John Winstead](https://github.com/jhnwnstd)