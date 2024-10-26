import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from utils.utils import load_config


class PCA_Transform(BaseEstimator, TransformerMixin):
    """Transformer class that fits PCA algoritm on input data replaces selected columns
    from the input dataset (pca_columns) with surogate pca columns PC1, PC2, ..., PCN
    that explain variance of the column to the level equal or higher than the threshold.

    Args:
    ----
        pca_threshold (float): how much variance must be explained by surogate columns
        all_columns   (list[string]): list of all column names in the dataset - required to
                                    initialize names of the output dataframe
        pca_columns:   (list[string]): list of callumns that will be replaced with PCA

    """

    def __init__(self, all_columns):
        config = load_config()
        self.pca_threshold = config["sources"]["apartments"]["pca"]["threshold"]
        self.pca_columns = ["num_cols__" + col for col in config["sources"]["apartments"]["pca"]["columns"]]
        self.pca = PCA(self.pca_threshold)
        self.all_columns = all_columns

    def fit(self, X, y=None):
        Xpd = pd.DataFrame(data=X, columns=self.all_columns)
        X_pca = Xpd.loc[:, self.pca_columns]
        self.pca.fit(X_pca)
        return self

    def transform(self, X, y=None):
        Xpd = pd.DataFrame(data=X, columns=self.all_columns)
        X_pca = Xpd.loc[:, self.pca_columns]
        X_nopca = Xpd.drop(self.pca_columns, axis=1)
        X_pca_transformed = self.pca.transform(X_pca)
        X_final = pd.concat(
            [
                X_nopca,
                pd.DataFrame(
                    data=X_pca_transformed, columns=[f"PC{i+1}" for i in range(len(self.pca.explained_variance_ratio_))],
                ),
            ],
            axis=1,
        )
        X_final.reset_index(drop=True, inplace=True)
        return pd.DataFrame(X_final)  
