import pandas as pd
import numpy as np

def get_feature_importance(model, preprocessor):
    """
    Extract feature importance from tree-based models
    """

    if not hasattr(model, "feature_importances_"):
        return None

    # Get feature names after preprocessing
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            encoded_cols = encoder.get_feature_names_out(cols)
            feature_names.extend(encoded_cols)

    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    return df.head(15)
