import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder


class defaultPredictor:
    def __init__(self, initial_request):
        self.request = initial_request
        self.encoder_directory = "./model/OneHotEncoderMemory.pickle"
        self.model_directory = "./model/lightgbm.pickle"
        self.column_dictionary = {
            "SEX": "GENDER",
            "EDUCATION": "EDU",
            "MARRIAGE": "MAR",
            "AgeBin": "AGE",
            "PAY_1": "PAY_1",
            "PAY_2": "PAY_2",
            "PAY_3": "PAY_3",
            "PAY_4": "PAY_4",
            "PAY_5": "PAY_5",
            "PAY_6": "PAY_6",
            "SE_MA": "SEMA",
            "SE_AG": "SEAG",
        }
        self.drop_columns = [
            "ID",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            "PAY_1",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "SE_MA",
            "SE_AG",
            "AgeBin",
            "GENDER_2",
            "default payment next month",
        ]

    def get_one_hot_enc(self, feature_col, enc, incomingDf, name):
        """
        maps an unseen column feature using one-hot-encoding previously fit against training data
        returns: a pd.DataFrame of newly one-hot-encoded feature
        """
        # assert isinstance(feature_col, pd.Series)
        # assert isinstance(enc, OneHotEncoder)

        # get unseen vector
        unseen_vector = feature_col.values.reshape(-1, 1)

        # convert onehot coding to the same column name.
        my_encoded_df = pd.DataFrame(
            enc.transform(unseen_vector).toarray(),
            columns=enc.get_feature_names([name]),
        )

        # merge new hotcoded data frame to previous dataframe.
        encoded_df = pd.concat([incomingDf, my_encoded_df], axis=1)

        return encoded_df

    def education_label_handler(self):

        # education
        if self.request["EDUCATION"] == 0:
            self.request["EDUCATION"] = 4
        elif self.request["EDUCATION"] == 6:
            self.request["EDUCATION"] = 5

    def marital_label_handler(self):

        if self.request["MARRIAGE"] == 0:
            self.request["MARRIAGE"] = 3

    def gender_marital_label_handler(self):

        if self.request["SEX"] == 1 and self.request["MARRIAGE"] == 1:  # married man
            self.request["SE_MA"] = 1
        elif self.request["SEX"] == 1 and self.request["MARRIAGE"] == 2:  # single man
            self.request["SE_MA"] = 2
        elif self.request["SEX"] == 1 and self.request["MARRIAGE"] == 3:  # divorced man
            self.request["SE_MA"] = 3
        elif (
            self.request["SEX"] == 2 and self.request["MARRIAGE"] == 1
        ):  # married woman
            self.request["SE_MA"] = 4
        elif self.request["SEX"] == 2 and self.request["MARRIAGE"] == 2:  # single woman
            self.request["SE_MA"] = 5
        elif (
            self.request["SEX"] == 2 and self.request["MARRIAGE"] == 3
        ):  # divorced woma
            self.request["SE_MA"] = 6

    def age_bin_label_handler(self):

        if self.request["AGE"] > 20 and self.request["AGE"] < 30:
            self.request["AgeBin"] = 1
        elif self.request["AGE"] >= 30 and self.request["AGE"] < 40:
            self.request["AgeBin"] = 2
        elif self.request["AGE"] >= 40 and self.request["AGE"] < 50:
            self.request["AgeBin"] = 3
        elif self.request["AGE"] >= 50 and self.request["AGE"] < 60:
            self.request["AgeBin"] = 4
        elif self.request["AGE"] >= 60 and self.request["AGE"] < 81:
            self.request["AgeBin"] = 5

    def gender_age_label_handler(self):

        if self.request["SEX"] == 1 and self.request["AgeBin"] == 1:
            self.request["SE_AG"] = 1
        elif self.request["SEX"] == 1 and self.request["AgeBin"] == 2:
            self.request["SE_AG"] = 2
        elif self.request["SEX"] == 1 and self.request["AgeBin"] == 3:
            self.request["SE_AG"] = 3
        elif self.request["SEX"] == 1 and self.request["AgeBin"] == 4:
            self.request["SE_AG"] = 4
        elif self.request["SEX"] == 1 and self.request["AgeBin"] == 5:
            self.request["SE_AG"] = 5
        elif self.request["SEX"] == 2 and self.request["AgeBin"] == 1:
            self.request["SE_AG"] = 6
        elif self.request["SEX"] == 2 and self.request["AgeBin"] == 2:
            self.request["SE_AG"] = 7
        elif self.request["SEX"] == 2 and self.request["AgeBin"] == 3:
            self.request["SE_AG"] = 8
        elif self.request["SEX"] == 2 and self.request["AgeBin"] == 4:
            self.request["SE_AG"] = 9
        elif self.request["SEX"] == 2 and self.request["AgeBin"] == 5:
            self.request["SE_AG"] = 10

    def closeness_handler(self):

        for i in range(6, 0, -1):
            self.request["Closeness_" + str(i)] = (
                self.request["LIMIT_BAL"] - self.request["BILL_AMT" + str(i)]
            ) / self.request["LIMIT_BAL"]

    def function_converter(self):

        self.education_label_handler()
        self.marital_label_handler()
        self.gender_marital_label_handler()
        self.age_bin_label_handler()
        self.gender_age_label_handler()
        self.closeness_handler()

        # convert request to dataframe
        incomingDf = pd.DataFrame(data=self.request, index=[0])

        # call encoder from folder
        encoder_dict = pickle.load(open(self.encoder_directory, "rb"))

        # convert to onehot encoder
        for key, value in self.column_dictionary.items():

            incomingDf = self.get_one_hot_enc(
                incomingDf[key], encoder_dict["encoder" + "_" + key], incomingDf, value
            )

        # drop unused columns
        incomingDf.drop(self.drop_columns, axis=1, inplace=True)

        return incomingDf

    def model_prediction(self, input):
        # ./model/lightgbm.pickle"
        lightgbm_tuned_mode = pickle.load(open(self.model_directory, "rb"))

        y_pred_test = (lightgbm_tuned_mode.predict_proba(input)[:, 1] >= 0.195).astype(
            bool
        )
        if y_pred_test == True:
            prediction = "Default"
        else:
            prediction = "Non-Default"

        return prediction


if __name__ == "__main__":
    # print(function_converter(initial).shape)
    # model_location = "./model/lightgbm.pickle"
    defaultInstance = defaultPredictor(initial)
    output = defaultInstance.function_converter()
    y_pred_test = defaultInstance.model_prediction(output)

    print(y_pred_test)


# input = {
#     "LIMIT_BAL": 20000,
#     "BILL_AMT1": 3913,
#     "BILL_AMT2": 3102,
#     "BILL_AMT3": 689,
#     "BILL_AMT4": 0,
#     "BILL_AMT5": 0,
#     "BILL_AMT6": 0,
#     "PAY_AMT1": 0,
#     "PAY_AMT2": 689,
#     "PAY_AMT3": 0,
#     "PAY_AMT4": 0,
#     "PAY_AMT5": 0,
#     "PAY_AMT6": 0,
#     "Default": 1,
#     "Closeness_6": 1.0,
#     "Closeness_5": 1.0,
#     "Closeness_4": 1.0,
#     "Closeness_3": 0.96555,
#     "Closeness_2": 0.8449,
#     "Closeness_1": 0.80435,
#     "GENDER_1": 0,
#     "EDU_1": 0,
#     "EDU_2": 1,
#     "EDU_3": 0,
#     "EDU_4": 0,
#     "EDU_5": 0,
#     "MAR_1": 1,
#     "MAR_2": 0,
#     "MAR_3": 0,
#     "AGE_1": 1,
#     "AGE_2": 0,
#     "AGE_3": 0,
#     "AGE_4": 0,
#     "AGE_5": 0,
#     "PAY_1_-2": 0,
#     "PAY_1_-1": 0,
#     "PAY_1_0": 0,
#     "PAY_1_1": 0,
#     "PAY_1_2": 1,
#     "PAY_1_3": 0,
#     "PAY_1_4": 0,
#     "PAY_1_5": 0,
#     "PAY_1_6": 0,
#     "PAY_1_7": 0,
#     "PAY_1_8": 0,
#     "PAY_2_-2": 0,
#     "PAY_2_-1": 0,
#     "PAY_2_0": 0,
#     "PAY_2_1": 0,
#     "PAY_2_2": 1,
#     "PAY_2_3": 0,
#     "PAY_2_4": 0,
#     "PAY_2_5": 0,
#     "PAY_2_6": 0,
#     "PAY_2_7": 0,
#     "PAY_2_8": 0,
#     "PAY_3_-2": 0,
#     "PAY_3_-1": 1,
#     "PAY_3_0": 0,
#     "PAY_3_1": 0,
#     "PAY_3_2": 0,
#     "PAY_3_3": 0,
#     "PAY_3_4": 0,
#     "PAY_3_5": 0,
#     "PAY_3_6": 0,
#     "PAY_3_7": 0,
#     "PAY_3_8": 0,
#     "PAY_4_-2": 0,
#     "PAY_4_-1": 1,
#     "PAY_4_0": 0,
#     "PAY_4_1": 0,
#     "PAY_4_2": 0,
#     "PAY_4_3": 0,
#     "PAY_4_4": 0,
#     "PAY_4_5": 0,
#     "PAY_4_6": 0,
#     "PAY_4_7": 0,
#     "PAY_4_8": 0,
#     "PAY_5_-2": 1,
#     "PAY_5_-1": 0,
#     "PAY_5_0": 0,
#     "PAY_5_2": 0,
#     "PAY_5_3": 0,
#     "PAY_5_4": 0,
#     "PAY_5_5": 0,
#     "PAY_5_6": 0,
#     "PAY_5_7": 0,
#     "PAY_5_8": 0,
#     "PAY_6_-2": 1,
#     "PAY_6_-1": 0,
#     "PAY_6_0": 0,
#     "PAY_6_2": 0,
#     "PAY_6_3": 0,
#     "PAY_6_4": 0,
#     "PAY_6_5": 0,
#     "PAY_6_6": 0,
#     "PAY_6_7": 0,
#     "PAY_6_8": 0,
#     "SEMA_1": 0,
#     "SEMA_2": 0,
#     "SEMA_3": 0,
#     "SEMA_4": 1,
#     "SEMA_5": 0,
#     "SEMA_6": 0,
#     "SEAG_1": 0,
#     "SEAG_2": 0,
#     "SEAG_3": 0,
#     "SEAG_4": 0,
#     "SEAG_5": 0,
#     "SEAG_6": 1,
#     "SEAG_7": 0,
#     "SEAG_8": 0,
#     "SEAG_9": 0,
#     "SEAG_10": 0
# }