from sklearn import preprocessing

"""
- label encoding
- one hot encoding
- binarizing
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.output_df = self.df.copy(deep=True)

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        else:
            raise Exception("Encoding type not understood")


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../input/train_cat.csv")
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)

    cat_feats = CategoricalFeatures(
        df, categorical_features=cols, encoding_type="label", handle_na=True

    )
    output_df = cat_feats.transform()
    print(output_df.head())


