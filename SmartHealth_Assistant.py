from collections import Counter, defaultdict
from math import log

def train_naive_bayes(training_data, feature_names, label_names):
    # Count occurrences of each label in training data
    label_counts = Counter(row[label_names] for row in training_data)
    total_rows = len(training_data)

    # Calculate prior probabilities for each label
    labels = sorted(label_counts.keys())
    priors = {}
    for label in labels:
        priors[label] = label_counts[label] / total_rows

    # Identify all possible values for each feature
    possible_feature_values = {feature: set() for feature in feature_names}
    for row in training_data:
        for feature in feature_names:
            possible_feature_values[feature].add(row[feature])

    # Calculate likelihood probabilities using Laplace smoothing
    likelihoods = {feature: defaultdict(dict) for feature in feature_names}
    for feature in feature_names:
        values = possible_feature_values[feature]

        # Calculate likelihood for each label
        for label in labels:
            # Filter rows matching current label
            subset = [row for row in training_data if row[label_names] == label]
            subset_count = len(subset)

            # Count occurrences of each feature value in subset
            value_counts = Counter(row[feature] for row in subset)

            # Calculate probability with Laplace smoothing (add 1 smoothing)
            for value in values:
                likelihoods[feature][label][value] = (
                    value_counts[value] + 1
                ) / (subset_count + len(values))
                
    return priors, likelihoods, labels, possible_feature_values


def predict_naive_bayes(case, priors, likelihoods, labels):
    # Initialize dictionary to store prediction scores for each label
    scores = {}     

    # Calculate log probability for each label
    for label in labels:
        # Start with log of prior probability
        score = log(priors[label])

        # Add log likelihood for each feature value
        for feature, value in case.items():
            score += log(likelihoods[feature][label][value])

        scores[label] = score
    
    # Return label with highest score
    predicted_label = max(scores, key=scores.get)
    
    return predicted_label, scores


def main():
    print("=" * 50)
    print("Clinical Symptom Risk Triage Assistant")
    print("=" * 50)

    training_data = [
        {"fever": "yes", "cough": "yes", "sneezing": "no", "body_aches": "yes", "condition": "flu"},
        {"fever": "yes", "cough": "yes", "sneezing": "no", "body_aches": "yes", "condition": "flu"},
        {"fever": "yes", "cough": "no", "sneezing": "no", "body_aches": "yes", "condition": "flu"},
        {"fever": "no", "cough": "no", "sneezing": "yes", "body_aches": "no", "condition": "allergy"},
        {"fever": "no", "cough": "yes", "sneezing": "yes", "body_aches": "no", "condition": "allergy"},
        {"fever": "no", "cough": "no", "sneezing": "yes", "body_aches": "no", "condition": "allergy"},
        {"fever": "no", "cough": "yes", "sneezing": "yes", "body_aches": "no", "condition": "allergy"},
        {"fever": "no", "cough": "yes", "sneezing": "no", "body_aches": "yes", "condition": "flu"},
    ]

    featue_names = ["fever", "cough", "sneezing", "body_aches"]
    label_names = "condition"

    priors, likelihoods, labels, possible_feature_values = train_naive_bayes(
        training_data, 
        featue_names, 
        label_names
    )

    print("\nClass Priors:")
    for label, value in priors.items():
        print(f" P({label}) = {value:.3f}")

    test_cases = [
        {"fever": "yes", "cough": "yes", "sneezing": "no", "body_aches": "yes"},
        {"fever": "no", "cough": "yes", "sneezing": "yes", "body_aches": "no"},
        {"fever": "yes", "cough": "yes", "sneezing": "yes", "body_aches": "no"},
    ]

    print("\nPredictions:")
    print("-" * 40)
    for i, case in enumerate(test_cases, start=1):
        prediction, scores = predict_naive_bayes(case, priors, likelihoods, labels)
        print(f"Test Case {i}")
        print("-" * 20)
        print(f"Symptoms: {case}")
        print(f"Predicted Condition: {prediction}")
        print("Log Scores:")
        for label, score in scores.items():
            print(f"  {label}: {score:.3f}")
        print("-" * 20)
        print ("\n")


if __name__ == "__main__":    
    main()