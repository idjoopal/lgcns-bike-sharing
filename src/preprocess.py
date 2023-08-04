import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


preprocess_pipeline = ColumnTransformer(
    transformers=[
        # TODO
        ("sqrt_transformer", FunctionTransformer(np.sqrt), ["humidity"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")