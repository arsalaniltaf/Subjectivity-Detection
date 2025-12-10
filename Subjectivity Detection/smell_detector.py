import csv
import os
import json
from textblob import TextBlob
from textstat import flesch_reading_ease
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from imblearn.over_sampling import SMOTE

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy English model...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class EnhancedSubjectivityDetector:
    def __init__(self, threshold=0.5, use_classifier=False, model_path=None):
        self.threshold = threshold
        self.use_classifier = use_classifier
        self.model_path = model_path
        self.classifier = None
        self.subjective_phrases = [
            'user friendly', 'easy to use', 'should be able to',
            'as needed', 'when necessary', 'if possible',
            'may', 'might', 'could', 'would', 'preferably',
            'if appropriate', 'as required', 'when needed',
            # Refined: Added RE-specific subjective terms from dataset analysis
            'safely', 'easy', 'easily', 'valid', 'automatic',
            'successful', 'external', 'periodic', 'sensitive',
            'another', 'part', 'case', 'unit', 'default',
            'effective'
        ]
        
        if use_classifier and model_path and os.path.exists(model_path):
            self.classifier = load(model_path)
            self.use_classifier = True

    def extract_features(self, text):
        blob = TextBlob(text)
        doc = nlp(text)
        
        # Count modal verbs
        modal_verbs = {'can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must'}
        modals = sum(1 for token in doc if token.text.lower() in modal_verbs)
        
        # Count hedges
        hedges = {'perhaps', 'maybe', 'possibly', 'likely', 'probably', 'sometimes', 'generally'}
        hedge_count = sum(1 for word in text.lower().split() if word in hedges)
        
        # Refined: Add counts for adjectives and adverbs (common in subjective RE language)
        adjectives = sum(1 for token in doc if token.pos_ == 'ADJ')
        adverbs = sum(1 for token in doc if token.pos_ == 'ADV')
        
        # Refined: Count -ly adverbs specifically (e.g., 'safely', 'easily')
        ly_adverbs = sum(1 for token in doc if token.pos_ == 'ADV' and token.text.lower().endswith('ly'))
        
        features = {
            'subjectivity': blob.sentiment.subjectivity,
            'polarity': blob.sentiment.polarity,
            'readability': flesch_reading_ease(text),
            'word_count': len(text.split()),
            'modal_verbs': modals,
            'hedges': hedge_count,
            'adjectives': adjectives,
            'adverbs': adverbs,
            'ly_adverbs': ly_adverbs,
            'has_subj_pattern': int(any(phrase in text.lower() for phrase in self.subjective_phrases))
        }
        
        return features

    def detect_with_classifier(self, text):
        if not self.classifier:
            raise ValueError("Classifier not loaded")
        
        # FIX: The classifier was trained on a list, so we must pass a list for prediction.
        features_list = list(self.extract_features(text).values())
        
        # Predict on a list of lists (a single inner list in this case)
        proba = self.classifier.predict_proba([features_list])[0]
        classification = self.classifier.predict([features_list])[0]
        return classification, proba[1]  # Return class and subjective probability

    def detect_with_rules(self, text):
        # Refined: Prioritize phrase matching with more RE-specific terms
        if any(phrase in text.lower() for phrase in self.subjective_phrases):
            return "Subjective", 1.0
        
        # Rule 2: Use TextBlob with threshold
        score = TextBlob(text).sentiment.subjectivity
        classification = "Subjective" if score >= self.threshold else "Objective"
        return classification, score

    def detect(self, text):
        if not text or not isinstance(text, str):
            return "Invalid", 0.0
        
        if self.use_classifier and self.classifier:
            return self.detect_with_classifier(text)
        else:
            return self.detect_with_rules(text)

def get_file_path(prompt_message):
    while True:
        file_path = input(prompt_message).strip()
        if not file_path:
            print("Error: Path cannot be empty.")
            continue
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            continue
        return file_path

def get_output_directory(prompt_message):
    while True:
        dir_path = input(prompt_message).strip()
        if not dir_path:
            print("Error: Path cannot be empty.")
            continue
        try:
            os.makedirs(dir_path, exist_ok=True)
            return dir_path
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")
            continue

def find_optimal_threshold(df):
    print("\nFinding optimal subjectivity threshold...")
    thresholds = [i/10 for i in range(1, 10)]
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = []
        for _, row in df.iterrows():
            score = TextBlob(row['Requirement']).sentiment.subjectivity
            pred = "Subjective" if score >= threshold else "Objective"
            preds.append(pred)
        
        report = classification_report(df['Subjectivity'].map(lambda x: "Subjective" if x == 1 else "Objective"), 
                                     preds, output_dict=True, zero_division=0)
        current_f1 = report['Subjective']['f1-score']
        print(f"Threshold {threshold:.1f}: F1 = {current_f1:.3f}")
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    
    print(f"\nOptimal threshold found: {best_threshold:.1f} (F1 = {best_f1:.3f})")
    return best_threshold

def train_classifier(df, output_dir):
    print("\nTraining custom classifier...")
    detector = EnhancedSubjectivityDetector()
    
    # Prepare features and labels
    features = []
    labels = []
    
    for _, row in df.iterrows():
        feat = detector.extract_features(row['Requirement'])
        features.append(list(feat.values()))
        labels.append("Subjective" if row['Subjectivity'] == 1 else "Objective")
    
    # Refined: Use stratified split to handle class imbalance (common in RE datasets)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Use SMOTE to oversample the minority class on the training data
    print("Applying SMOTE to balance the training data...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    print(f"Original training sample counts: {pd.Series(y_train).value_counts()}")
    print(f"Resampled training sample counts: {pd.Series(y_res).value_counts()}")
    
    # Refined: Remove class_weight since SMOTE already balances the data
    clf = RandomForestClassifier(n_estimators=100, random_state=47)
    clf.fit(X_res, y_res)
    
    # Evaluate
    preds = clf.predict(X_test)
    print("\nClassifier Performance:")
    print(classification_report(y_test, preds))
    
    # Save model
    model_path = os.path.join(output_dir, 'subjectivity_classifier.joblib')
    dump(clf, model_path)
    print(f"\nClassifier saved to {model_path}")
    
    return model_path

def generate_visualizations(df, output_dir):
    # Bar chart of label distribution
    label_counts = df['Detected_Subjectivity'].value_counts().to_dict()
    
    plt.figure(figsize=(8, 6))
    plt.bar(label_counts.keys(), label_counts.values(), color=['#36A2EB', '#FF6384'])
    plt.title("Distribution of Predicted Subjectivity Labels")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Count")
    chart_path = os.path.join(output_dir, 'label_distribution.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Label distribution chart saved to {chart_path}")
    
def generate_metrics_chart(report, output_dir):
    """
    Generates a bar chart showing precision, recall, and F1-score for weighted average only.
    """
    metrics = {
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score']
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    plt.figure(figsize=(10, 7))
    bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#2ca02c', '#ff7f0e'])
    
    # Add labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')
        
    plt.title('Evaluation Metrics for Subjectivity Detection (Weighted Average)')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'performance_metrics.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Performance metrics chart saved to {chart_path}")

def main():
    print("=== Enhanced Subjectivity Detection for Requirements (Refined Version) ===")
    
    # Get file paths
    input_path = get_file_path("Enter path to input CSV file (e.g., DS1.csv): ")
    output_dir = get_output_directory("Enter output directory path: ")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Check required columns
    if 'Requirement' not in df.columns:
        print("Error: Input file must contain 'Requirement' column")
        return
    
    has_labels = 'Subjectivity' in df.columns
    
    # Configuration
    print("\n=== Configuration ===")
    use_classifier = False
    model_path = None
    optimal_threshold = 0.5
    
    if has_labels:
        # Find optimal threshold if we have labels
        optimal_threshold = find_optimal_threshold(df)
        
        # Ask if user wants to train classifier
        train_clf = input("\nTrain a custom classifier? (y/n): ").lower() == 'y'
        if train_clf:
            model_path = train_classifier(df, output_dir)
            use_classifier = True
    
    # Initialize detector
    detector = EnhancedSubjectivityDetector(
        threshold=optimal_threshold if has_labels else 0.5,
        use_classifier=use_classifier,
        model_path=model_path
    )
    
    # Process requirements
    results = []
    for _, row in df.iterrows():
        req = row['Requirement']
        classification, score = detector.detect(req)
        
        result = {
            'Requirement': req,
            'Detected_Subjectivity': classification,
            'Subjectivity_Score': score,
            'Detector_Type': 'Classifier' if use_classifier else 'Rule-based'
        }
        
        if has_labels:
            result['True_Label'] = "Subjective" if row['Subjectivity'] == 1 else "Objective"
        
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'subjectivity_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Generate visualizations
    generate_visualizations(results_df, output_dir)
    
    # Calculate metrics if we have true labels
    if has_labels:
        y_true = results_df['True_Label']
        y_pred = results_df['Detected_Subjectivity']
        
        print("\n=== Final Performance ===")
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))
        
        # Generate the new metrics chart
        generate_metrics_chart(report, output_dir)
        
        # Save metrics
        metrics = {
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=['Objective', 'Subjective']).tolist(),
            'detector_config': {
                'type': 'Classifier' if use_classifier else 'Rule-based',
                'threshold': detector.threshold if not use_classifier else None
            }
        }
        
        metrics_path = os.path.join(output_dir, 'performance_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
