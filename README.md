**Enhanced Subjectivity Detection for Software Requirements
**Author: Muhammad Arsalan

This project implements an advanced Subjectivity Detection Tool for software
requirements using rule-based analysis, sentiment features, linguistic analysis,
readability metrics, and an optional machine learning classifier trained using
SMOTE-balanced Random Forest. The tool supports threshold optimization,
visualization, and detailed evaluation reporting.

======================================================================
1. PROJECT FEATURES
======================================================================
- Rule-based subjectivity detection using:
    * TextBlob polarity & subjectivity
    * Flesch Reading Ease readability score
    * spaCy linguistic features (POS, adverbs, adjectives, -ly adverbs)
    * Modal verbs detection
    * Hedge word detection
    * Phrase-level subjective patterns (RE-specific)
- Optional supervised classifier (RandomForest + SMOTE)
- Automatic threshold optimization (for rule-based mode)
- Evaluation with precision, recall, f1-score
- Confusion matrix & metrics saved as JSON
- Visualizations:
    * Distribution of predicted labels
    * Weighted precision/recall/F1 metrics chart
- Saves outputs to user-selected directory

======================================================================
2. DATASET REQUIREMENTS
======================================================================
Your input CSV MUST contain:

    Requirement   (string)     → requirement sentence
    Subjectivity  (0 or 1)     → label indicating subjective (1) or objective (0)

If these column names do NOT match exactly, the script will not run.
Rename your columns before running.

Example:
---------------------------------------------------------
Requirement,Subjectivity
"The system should ideally respond quickly",1
"The system logs user activity",0
---------------------------------------------------------

======================================================================
3. FILE STRUCTURE
======================================================================
project_folder/
│
├── subjectivity_detector.py       (main script)
├── DS1.csv / dataset.csv          (your dataset file)
└── output_folder/                 (generated automatically)

======================================================================
4. INSTALLATION REQUIREMENTS
======================================================================
Python 3.8 or newer recommended.

Required packages:
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    joblib
    textblob
    textstat
    spacy
    imblearn (for SMOTE)

Install using:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib textblob textstat imblearn spacy

Download spaCy English model:
    python -m spacy download en_core_web_sm

======================================================================
5. HOW THE TOOL WORKS
======================================================================

(A) RULE-BASED DETECTION
    - Extract features using TextBlob subjectivity, polarity, readability,
      modal verbs, hedges, adjectives, adverbs, -ly adverbs, and phrase patterns.
    - Use threshold to classify as Subjective or Objective.
    - Automatically optimizes threshold using labeled data (if provided).

(B) CLASSIFIER-BASED DETECTION (Optional)
    - Computes 10+ linguistic/numeric features per requirement.
    - Uses SMOTE to balance training data.
    - Trains RandomForestClassifier.
    - Evaluates classifier performance on test set.
    - Saves model and prediction results.

(C) VISUALIZATION
    - Saves bar chart of predicted label distribution.
    - Saves weighted precision/recall/F1 chart.

(D) EXPORTS
    - subjectivity_results.csv
    - performance_metrics.json
    - label_distribution.png
    - performance_metrics.png
    - subjectivity_classifier.joblib (if classifier mode used)

======================================================================
6. USAGE INSTRUCTIONS
======================================================================

MACOS:
    1. Open Terminal.
    2. Navigate to project folder:
           cd /path/to/project_folder
    3. Run the script:
           python3 subjectivity_detector.py
    4. Provide:
           - Path to dataset CSV (must contain Requirement & Subjectivity)
           - Output directory path

WINDOWS:
    1. Open CMD or PowerShell.
    2. Navigate to project folder:
           cd C:\path\to\project_folder
    3. Run the script:
           python subjectivity_detector.py
    4. Provide required paths when prompted.

======================================================================
7. OUTPUT FILES EXPLAINED
======================================================================
Inside the selected output directory, you will find:

subjectivity_results.csv        → predictions & scores for each requirement
performance_metrics.json        → confusion matrix + metrics
label_distribution.png          → visualization of predicted labels
performance_metrics.png         → weighted precision/recall/F1 chart
subjectivity_classifier.joblib  → saved model (only if classifier used)

======================================================================
8. ABOUT THE DATASET (ARTA REQUIREMENT SMELL DATASET)
======================================================================
This project can be applied on or extended with publicly available datasets such
as the dataset released with the paper:

"Requirement testability measurement based on requirement smells."

This dataset is published through Zenodo and contains real-world software
requirements annotated with various requirement smells (ambiguity, subjectivity,
unverifiability, vague phrasing, etc.). The dataset supports the development of
automated requirement-quality assessment tools and is part of the ARTA
(Automatic Requirement Testability Analyzer) project, which assists requirement
engineers in performing systematic quality assurance.

======================================================================
9. DATASET REFERENCE (CITATION)
======================================================================

Zenodo Citation:
Requirement testability measurement based on requirement smells – dataset and results.
Zenodo. DOI: 10.5281/zenodo.4266727
URL: https://zenodo.org/record/4266727

APA Format:
Femmer, H., Vogelsang, A., Eder, S., & Juergens, E. (2020).
Requirement testability measurement based on requirement smells – dataset and results.
Zenodo. https://doi.org/10.5281/zenodo.4266727

======================================================================
10. NOTES
======================================================================
- Only works if column names match exactly:
      Requirement
      Subjectivity
- To use classifier mode effectively, a labeled dataset is required.
- Rule-based mode does not require labels.
- SMOTE helps with class imbalance in RE datasets.
- Threshold optimization improves rule-based accuracy significantly.

======================================================================
11. CONTACT
======================================================================
For questions, feedback, or improvements:
Contact the author or open an issue on the repository.

