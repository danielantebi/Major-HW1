import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data(training_data, new_data):
    #creates a copy of new_data
    copy_df = new_data.copy()

    #fills NA data for columns with missing entries
    copy_df["household_income"] = copy_df.household_income.fillna(training_data.household_income.median())

    #makes special property according to blood_type feature
    SpecialProperty = copy_df["blood_type"].isin(["O+", "B+"])
    copy_df = copy_df.drop("blood_type", axis=1)
    copy_df["SpecialProperty"] = SpecialProperty

    #applying normalization for PCR features
    MinMaxScalarNormalization = ["PCR_03", "PCR_05", "PCR_06", "PCR_08", "PCR_10"]
    StandardScalarNormalization = ["PCR_01", "PCR_02", "PCR_04", "PCR_07", "PCR_09"]

    standard_scalar = StandardScaler()
    minmax_scalar = MinMaxScaler()

    standard_scalar.fit(training_data[StandardScalarNormalization])
    minmax_scalar.fit(training_data[MinMaxScalarNormalization])

    copy_df[MinMaxScalarNormalization] = minmax_scalar.transform(copy_df[MinMaxScalarNormalization])
    copy_df[StandardScalarNormalization] = standard_scalar.transform(copy_df[StandardScalarNormalization])

    #data is prepared
    return copy_df
