# src/features/naming.py

def build_flat_feature_names(feature_cols, window_size):
    flat_feature_names = []

    for step in range(window_size):
        lag = window_size - step
        for feat in feature_cols:
            flat_feature_names.append(f"{feat}_t-{lag}")

    return flat_feature_names