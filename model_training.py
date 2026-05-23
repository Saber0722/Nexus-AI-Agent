from sklearn.ensemble import RandomForestClassifier

def initialize_random_forest_classifier(n_estimators=100, max_depth=None, random_state=42):
    """
    Initialize a Random Forest classifier with specified parameters.

    Parameters:
        n_estimators (int): The number of trees in the forest.
        max_depth (int or None): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        random_state (int): Controls both the randomness of the bootstrapping of the samples used when building trees (if `bootstrap=True`) and the randomness of the splits made during the tree induction.

    Returns:
        RandomForestClassifier: An initialized Random Forest classifier instance.
    """
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    return classifier