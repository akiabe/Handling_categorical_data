from sklearn import preprocessing

"""
- label encoding
- one hot encoding
- binarization
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")

        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.output_df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()

        elif self.enc_type == "binary":
            return self._label_binarization()

        elif self.enc_type == "ohe":
            return self._one_hot()

        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)

        else:
            raise Exception("Encoding type not understood!")

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../input/train_cat.csv")#.head(500)
    df_test = pd.read_csv("../input/test.csv")#.head(500)

    #train_idx = df["id"].values
    #test_idx = df_test["id"].values

    train_len = len(df)
    test_len = len(df_test)

    df_test["target"] = -1
    full_data = pd.concat([df, df_test])

    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats = CategoricalFeatures(
        full_data, categorical_features=cols, encoding_type="ohe", handle_na=True
    )
    full_data_transformd = cat_feats.fit_transform()
    #print(full_data_transform)

    #train_df = full_data_transformd[full_data_transformd["id"].isin(train_idx)].reset_index(drop=True)
    #test_df = full_data_transformd[full_data_transformd["id"].isin(test_idx)].reset_index(drop=True)

    train_df = full_data_transformd[:train_len, :]
    test_df = full_data_transformd[train_len:, :]

    print(train_df.shape)
    print(test_df.shape)

