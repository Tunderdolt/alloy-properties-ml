# main

from typing import Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
from scipy import optimize
from sklearn.metrics import r2_score

# Define fixed values as Literals
Element = Literal[
    "fe", "c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"
]
Property = Literal["yield_strength", "tensile_strength", "elongation"]
ElongationCategory = Literal["weak", "medium", "strong", "NaN"]

# Define constants
ELEMENTS: list[Element] = list(get_args(Element))
PROPERTIES: list[Property] = list(get_args(Property))
ELEMENT_WEIGHT: dict[str, float] = {
    "fe": 55.845,
    "c": 12.011,
    "mn": 54.938,
    "si": 28.086,
    "cr": 51.996,
    "ni": 58.693,
    "mo": 95.94,
    "v": 50.942,
    "n": 14.007,
    "nb": 92.906,
    "co": 58.933,
    "w": 183.84,
    "al": 26.982,
    "ti": 47.867,
}


class DataframeWriter:
    """
    A class which Writes data to a DataFrame

    Atributes
    ---------
    data : pd.DataFrame
        The DataFrame that will be written to

    Methods
    -------
    atom_to_weight_percent(formula : str)
        Takes a chemical formula and converts it to a weight percent formula

    data_fill()
        Introduce calculated weight percents into data

    data_shuffler(data : pd.DataFrame, n : int)
        randomly sort data into three stratified samples
    """

    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame that will be written to
        """
        self.data = data

    def atom_to_weight_percent(self, formula: str):
        """Returns a dictionary of weight percent relating to an element from a chemical formula provided as a continuous string

        Parameters
        ----------
        formula : str
            A continuous string denoting the chemical formula in the form element, ratio without delimeters

        Returns
        -------
        weight_dict : dict[str, float]
            A dictionary containing the element and it's respective weight% omposition
        """
        atom_list: list[str] = list(formula)

        flip: bool = False
        joined_list: list = []
        k: int = 0

        # The following loop splits the continuous string which has been converted to a list by
        # element and composition and puts those respectively into a list as single elements.
        # Regex can be used but I felt it was unclear so did this manually instead.
        for i in range(1, len(atom_list)):
            if not flip:
                k += 1
                if not atom_list[i].isalpha():
                    flip = True

                    if k == 1:
                        joined_list.append(atom_list[i - 1])

                    elif k == 2:
                        joined_list.append(atom_list[i - 2] + atom_list[i - 1])

                    k = 0

            elif flip:
                k += 1
                if atom_list[i].isalpha():
                    flip = False

                    intermediate = ""
                    for j in range(i - k, i):
                        intermediate += atom_list[j]
                    joined_list.append(intermediate)

                    k = 0

                elif i == len(atom_list) - 1:
                    intermediate = ""
                    for j in range(i - k, i + 1):
                        intermediate += atom_list[j]
                    joined_list.append(intermediate)

        formula_tuple_list: list[tuple[str, float]] = []

        # Converts list of strings into a list of tuples in the form (str, float)
        for i in range(0, len(joined_list), 2):
            formula_tuple_list.append(
                tuple((joined_list[i], float(joined_list[i + 1])))
            )

        formula_dict: dict[str, float] = dict(formula_tuple_list)
        weight_dict_intermediate: dict[str, float] = {}

        # Converts atom% to weight% from the chemical formula
        for key in formula_dict:
            weight_dict_intermediate[key] = formula_dict[key] * ELEMENT_WEIGHT[key]

        total_weight = sum(weight_dict_intermediate.values())

        weight_dict: dict[str, float] = {}
        for key in weight_dict_intermediate:
            weight_dict[key] = round(
                100 * weight_dict_intermediate[key] / total_weight, 2
            )

        return weight_dict

    def data_fill(self):
        """Inserts new columns into data that are filled with manually calculated weight percents

        Returns
        -------
        self.data : pd.DataFrame
            The modified attribute DataFrame
        """
        fe_calc: list = []
        c_calc: list = []
        mn_calc: list = []
        si_calc: list = []
        cr_calc: list = []
        ni_calc: list = []
        mo_calc: list = []
        v_calc: list = []
        n_calc: list = []
        nb_calc: list = []
        co_calc: list = []
        w_calc: list = []
        al_calc: list = []
        ti_calc: list = []

        # Creates lists for each element's calculated weight%
        for i in range(0, len(self.data.index)):
            weight_dict: dict = self.atom_to_weight_percent(self.data["formula"][i])
            fe_calc.append(weight_dict.get("fe", 0))
            c_calc.append(weight_dict.get("c", 0))
            mn_calc.append(weight_dict.get("mn", 0))
            si_calc.append(weight_dict.get("si", 0))
            cr_calc.append(weight_dict.get("cr", 0))
            ni_calc.append(weight_dict.get("ni", 0))
            mo_calc.append(weight_dict.get("mo", 0))
            v_calc.append(weight_dict.get("v", 0))
            n_calc.append(weight_dict.get("n", 0))
            nb_calc.append(weight_dict.get("nb", 0))
            co_calc.append(weight_dict.get("co", 0))
            w_calc.append(weight_dict.get("w", 0))
            al_calc.append(weight_dict.get("al", 0))
            ti_calc.append(weight_dict.get("ti", 0))

        # Adds column to data for each calculated composition
        self.data["fe_calc"] = fe_calc
        self.data["c_calc"] = c_calc
        self.data["mn_calc"] = mn_calc
        self.data["si_calc"] = si_calc
        self.data["cr_calc"] = cr_calc
        self.data["ni_calc"] = ni_calc
        self.data["mo_calc"] = mo_calc
        self.data["v_calc"] = v_calc
        self.data["n_calc"] = n_calc
        self.data["nb_calc"] = nb_calc
        self.data["co_calc"] = co_calc
        self.data["w_calc"] = w_calc
        self.data["al_calc"] = al_calc
        self.data["ti_calc"] = ti_calc

        return self.data

    def data_shuffler(self, n: int):
        """Splits data into n number of stratified sample DataFrames
        Samples are stratified by choosing randomly from samples seperated by iron composition

        Parameters
        ----------
        n : int
            The number of samples

        Returns
        -------
        fe_sample_data : list[pd.DataFrame]
            List of stratified sample DataFrames
        """
        sorted_data: pd.DataFrame = self.data.sort_values(["fe"], ignore_index=True)
        length_of_segment = int(round(len(self.data.index) / n))
        shuffled_fe_dict: dict[str, pd.DataFrame] = {}

        # Fills dictionary with random samples split by iron percent
        for i in range(0, n - 1):
            shuffled_fe_dict[f"shuffled_fe_data{i+1}"] = (
                sorted_data.iloc[i * length_of_segment : (i + 1) * length_of_segment]
                .sample(frac=1)
                .reset_index(drop=True)
            )

        # Fills final sample in dictionary for the possibility that the number of rows of the DataFrame is not perfectly divisible by n
        shuffled_fe_dict[f"shuffled_fe_data{n}"] = (
            sorted_data.iloc[(n - 1) * length_of_segment : len(self.data.index)]
            .sample(frac=1)
            .reset_index(drop=True)
        )

        fe_sample_data: list[pd.DataFrame] = []
        partial_sample_dict: dict[str, pd.DataFrame] = {}

        # Seperates shuffled samples into random samples by choosing first samples present in the shuffled samples.
        # This will be the first 1/n of each shuffled sample.
        for i in range(1, n):
            sample_df: pd.DataFrame = pd.DataFrame([])
            for j in range(1, n + 1):
                partial_sample_dict[f"partial_sample{j}"] = shuffled_fe_dict[
                    f"shuffled_fe_data{j}"
                ].iloc[0 : int(round(length_of_segment / n))]
                sample_df = pd.concat(
                    [sample_df, partial_sample_dict[f"partial_sample{j}"]]
                )
                shuffled_fe_dict[f"shuffled_fe_data{j}"] = (
                    shuffled_fe_dict[f"shuffled_fe_data{j}"]
                    .reset_index(drop=True)
                    .drop(np.arange(0, int(round(length_of_segment / n))))
                )

            # Places sample daataframes indside a list
            fe_sample_data.append(sample_df.reset_index(drop=True))

        sample_df = pd.DataFrame([])

        # Fills final sample in list for the possibility that the number of rows in the shuffled samples is not perfectly divisible by n
        for i in range(1, n + 1):
            sample_df = pd.concat([sample_df, shuffled_fe_dict[f"shuffled_fe_data{i}"]])

        fe_sample_data.append(sample_df.reset_index(drop=True))

        return fe_sample_data


def a_calc(data: pd.DataFrame, y: list[str] = PROPERTIES, x: list[str] = ELEMENTS):
    """Returns coefficient of a linear fit (y = Ax) from a DataFrame assuming no intercept

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame which contains the data to fit to
    x : list[str]
        The columns in data which correspond to the x vector
    y : list[str]
        The columns in data which correspond to the y vector

    Returns
    -------
    A : np.ndarray
        Coefficient of linear fit
    """
    # Remove rows that contain instances of np.nan as they cannot contribute to a fit as they are not plottable
    data = data.dropna()
    composition_vectors = data[[element for element in x]].to_numpy()
    properties_vectors = data[[properties for properties in y]].to_numpy()

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(composition_vectors, properties_vectors)

    A: np.ndarray = model.coef_

    return A


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate mean average percentage error for a set of data

    Parameters
    ----------
    actual : np.ndarray
        The value(s) we know to be true
    predicted : np.ndarray
        The values(s) we calculate

    Returns
    -------
    Mean Average Percentage Error : np.ndarray
        The Percentage Error for a linear fit of data
    """
    return np.mean(np.abs((actual - predicted) / actual), axis=0) * 100


def calculate_std(actual: np.ndarray, predicted: np.ndarray, mean: float = 0):
    """Calculate Standard Deviation for a set of data

    Parameters
    ----------
    actual : np.ndarray
        The value(s) we know to be true
    predicted : np.ndarray
        The values(s) we calculate
    mean : float, optional
        The expected mean difference of the actual and predicted values, default = 0

    Returns
    -------
    Standard Deviation : np.ndarray
        The Standard Deviation for the data
    """
    return np.sqrt(
        np.sum(np.square((actual - predicted) - mean), axis=0)
        / (actual - predicted).shape[0]
    )


def categorise_elongation(elongation: float) -> ElongationCategory:
    """Catagorises elongation by fragile, medium, and strong dependent on the value

    Parameters
    ----------
    elongation : float
        The value of elongation to be catagorised

    Returns
    -------
    Catagory : str
        The catagory which the value falls into
    """
    # Cannot classify a value that does not exist, i.e is np.nan
    if np.isnan(elongation):
        return np.nan
    if elongation < 5:
        return "fragile"
    if elongation > 10:
        return "strong"
    return "medium"


alloy_properties = pd.read_csv(
    r"C:\Users\sambi\Programming\alloy-properties-ml\database_steel_properties.csv",
    skiprows=1,
)
# Changes all chemical formulas to lowercase
alloy_properties["formula"] = alloy_properties["formula"].apply(
    lambda value: value.lower()
)

dataframe_writer = DataframeWriter(alloy_properties)

alloy_properties: pd.DataFrame = dataframe_writer.data_fill()

# Concatanates weight percents to replace instances np.nan in composition columns
refined_alloy_properties: pd.DataFrame = pd.DataFrame(
    {
        element: alloy_properties[element].combine_first(
            alloy_properties[f"{element}_calc"]
        )
        for element in ELEMENTS
        # No column present for fe so ignore
        if element != "fe"
    }
)
# Fills the remianing data within refined_alloy_proerties.
# We generate this new DataFrame as it contains better data to work with
refined_alloy_properties["fe"] = alloy_properties["fe_calc"]
refined_alloy_properties["combined_compositions"] = refined_alloy_properties[
    [element for element in ELEMENTS]
].values.tolist()
refined_alloy_properties["yield_strength"] = alloy_properties["yield strength"]
refined_alloy_properties["tensile_strength"] = alloy_properties["tensile strength"]
refined_alloy_properties["elongation"] = alloy_properties["elongation"]
refined_alloy_properties["combined_properties"] = refined_alloy_properties[
    [prop for prop in PROPERTIES]
].values.tolist()
refined_alloy_properties["combined_properties"] = refined_alloy_properties[
    "combined_properties"
].apply(lambda prop: np.array(prop))

test_alloy_properties = refined_alloy_properties

A_learned = a_calc(refined_alloy_properties)

# Converts lists of compositions to ndarrays
composition_vectors = refined_alloy_properties[
    [element for element in ELEMENT_WEIGHT.keys()]
].to_numpy()

# Creates columns of predicted mechanical properties
# Elongation is seperated to easier catagorise it
refined_alloy_properties["elongation_predicted"] = refined_alloy_properties[
    "combined_compositions"
].apply(lambda prop: np.matmul(A_learned, prop)[2])
refined_alloy_properties["combined_predicted"] = refined_alloy_properties[
    "combined_compositions"
].apply(lambda prop: np.matmul(A_learned, prop))

# Removes rows with instances of np.nan
reduced_alloy_properties = refined_alloy_properties.dropna()

# Creates arrays containing data used for validation
properties_array = np.array(
    reduced_alloy_properties["combined_properties"].values.tolist()
)
predicted_array = np.array(
    reduced_alloy_properties["combined_predicted"].values.tolist()
)

# Calculates R^2 value for each property
r_squared = list(
    r2_score(
        properties_array,
        predicted_array,
        multioutput="raw_values",
    )
)

# Calculates standard deviation for each property
std_combined = calculate_std(properties_array, predicted_array)

# Calculates Mean Average Percentage Error for each property
mape_combined = calculate_mape(properties_array, predicted_array)

# Catagorises elongation and inserts this as a column into the DataFrame
refined_alloy_properties["elongation_catagorised_true"] = refined_alloy_properties[
    "elongation"
].apply(categorise_elongation)
refined_alloy_properties["elongation_catagorised_predicted"] = refined_alloy_properties[
    "elongation_predicted"
].apply(categorise_elongation)

# The following finds the precentage of incorrect calculates catagorisations for elongation
count_have_data = 0
count_not_equal = 0

for i in range(0, len(refined_alloy_properties.index)):
    if refined_alloy_properties.loc[i, "elongation_catagorised_true"] == "NaN":
        continue
    count_have_data += 1
    if (
        refined_alloy_properties.loc[i, "elongation_catagorised_true"]
        != refined_alloy_properties.loc[i, "elongation_catagorised_predicted"]
    ):
        count_not_equal += 1

# Print the errors that have been calculated in the form [yield strength, tesnile strength, elongation]
print(f"The R^2 for the entire dataset is {r_squared}")
print(f"The std for the entire dataset is {std_combined}")
print(f"The MAPE for the entire dataset is {mape_combined}")
print(
    f"The percentage of incorrect assignments of elongation is {count_not_equal / count_have_data}"
)

refined_alloy_properties.to_csv(
    r"C:\Users\sambi\Programming\alloy-properties-ml\modified_steel_properties_database.csv"
)

#
# The following is to see if there is a correlation between error and and element composition
#

## Creates columns for percentage error for each property
#refined_alloy_properties["percent_error_combined"] = abs(
#    (
#        refined_alloy_properties["combined_properties"]
#        - refined_alloy_properties["combined_predicted"]
#    )
#    / refined_alloy_properties["combined_properties"]
#)
#
## Seperates those into their respective columns
#refined_alloy_properties["percent_error_ys"] = refined_alloy_properties[
#    "percent_error_combined"
#].apply(lambda prop: prop[0])
#refined_alloy_properties["percent_error_ts"] = refined_alloy_properties[
#    "percent_error_combined"
#].apply(lambda prop: prop[1])
#refined_alloy_properties["percent_error_e"] = refined_alloy_properties[
#    "percent_error_combined"
#].apply(lambda prop: prop[2])
#refined_alloy_properties = refined_alloy_properties.dropna()
#
## The following plots all plot a distribution of percentage error of a particular property
## These are explained and descirbed further in the text document
#figure, axis = plt.subplots(2, 7)
#
#
#plots_ys: dict[str, plt.Axes] = {}
#k = 0
#
## Loop generates the plots for each element
#for i in range(0, len(ELEMENTS)):
#    j = i
#    if j >= int(len(ELEMENTS) / 2):
#        k = 1
#        j = i - int(len(ELEMENTS) / 2)
#    plots_ys[f"{ELEMENTS[i]}_err"] = sns.kdeplot(
#        data=refined_alloy_properties,
#        y=f"{ELEMENTS[i]}",
#        x="percent_error_ys",
#        ax=axis[k, j],
#        fill=True,
#        cmap="rocket_r",
#    )
#    plots_ys[f"{ELEMENTS[i]}_err"].set_xlim([0, 0.6])
#    plots_ys[f"{ELEMENTS[i]}_err"].set_ylim(bottom=0)
#
## Sets ylims for each plot, this was decided by looking at the graphs after they have been plotted
#plots_ys["fe_err"].set_ylim([60, 85])
#plots_ys["c_err"].set_ylim(top=0.45)
#plots_ys["mn_err"].set_ylim(top=1)
#plots_ys["si_err"].set_ylim(top=2.5)
#plots_ys["cr_err"].set_ylim(top=22.5)
#plots_ys["ni_err"].set_ylim(top=25)
#plots_ys["mo_err"].set_ylim(top=8)
#plots_ys["v_err"].set_ylim(top=1.4)
#plots_ys["n_err"].set_ylim(top=0.055)
#plots_ys["nb_err"].set_ylim(top=0.25)
#plots_ys["co_err"].set_ylim(top=22.5)
#plots_ys["w_err"].set_ylim(top=2.5)
#plots_ys["al_err"].set_ylim(top=1.4)
#plots_ys["ti_err"].set_ylim(top=2.75)
#
## Seperates each subplot such that overlap of subplots and axes titles won't be present
#plt.tight_layout(w_pad=-2.5, h_pad=-1)
#plt.show()
#
#figure, axis = plt.subplots(2, 7)
#
#plots_ts: dict[str, plt.Axes] = {}
#k = 0
#
## Loop generates the plots for each element
#for i in range(0, len(ELEMENTS)):
#    j = i
#    if j >= int(len(ELEMENTS) / 2):
#        k = 1
#        j = i - int(len(ELEMENTS) / 2)
#    plots_ts[f"{ELEMENTS[i]}_err"] = sns.kdeplot(
#        data=refined_alloy_properties,
#        y=f"{ELEMENTS[i]}",
#        x="percent_error_ts",
#        ax=axis[k, j],
#        fill=True,
#        cmap="rocket_r",
#    )
#    plots_ts[f"{ELEMENTS[i]}_err"].set_xlim([0, 0.6])
#    plots_ts[f"{ELEMENTS[i]}_err"].set_ylim(bottom=0)
#
## Sets ylims for each plot, this was decided by looking at the graphs after they have been plotted
#plots_ts["fe_err"].set_ylim([60, 85])
#plots_ts["c_err"].set_ylim(top=0.5)
#plots_ts["mn_err"].set_ylim(top=1.1)
#plots_ts["si_err"].set_ylim(top=2.5)
#plots_ts["cr_err"].set_ylim(top=22.5)
#plots_ts["ni_err"].set_ylim(top=25)
#plots_ts["mo_err"].set_ylim(top=8)
#plots_ts["v_err"].set_ylim(top=1.4)
#plots_ts["n_err"].set_ylim(top=0.055)
#plots_ts["nb_err"].set_ylim(top=0.25)
#plots_ts["co_err"].set_ylim(top=22.5)
#plots_ts["w_err"].set_ylim(top=2.75)
#plots_ts["al_err"].set_ylim(top=1.4)
#plots_ts["ti_err"].set_ylim(top=2.75)
#
## Seperates each subplot such that overlap of subplots and axes titles won't be present
#plt.tight_layout(w_pad=-2.5, h_pad=-1)
#plt.show()
#
#figure, axis = plt.subplots(2, 7)
#
#plots_e: dict[str, plt.Axes] = {}
#k = 0
#
## Loop generates the plots for each element
#for i in range(0, len(ELEMENTS)):
#    j = i
#    if j >= int(len(ELEMENTS) / 2):
#        k = 1
#        j = i - int(len(ELEMENTS) / 2)
#    plots_e[f"{ELEMENTS[i]}_err"] = sns.kdeplot(
#        data=refined_alloy_properties,
#        y=f"{ELEMENTS[i]}",
#        x="percent_error_e",
#        ax=axis[k, j],
#        fill=True,
#        cmap="cividis_r",
#    )
#    plots_e[f"{ELEMENTS[i]}_err"].set_xlim([0, 2.5])
#    plots_e[f"{ELEMENTS[i]}_err"].set_ylim(bottom=0)
#
## Sets ylims for each plot, this was decided by looking at the graphs after they have been plotted
#plots_e["fe_err"].set_ylim([60, 85])
#plots_e["c_err"].set_ylim(top=0.5)
#plots_e["mn_err"].set_ylim(top=1.2)
#plots_e["si_err"].set_ylim(top=2.5)
#plots_e["cr_err"].set_ylim(top=22.5)
#plots_e["ni_err"].set_ylim(top=25)
#plots_e["mo_err"].set_ylim(top=8)
#plots_e["v_err"].set_ylim(top=2.25)
#plots_e["n_err"].set_ylim(top=0.06)
#plots_e["nb_err"].set_ylim(top=0.36)
#plots_e["co_err"].set_ylim(top=22.5)
#plots_e["w_err"].set_ylim(top=2.25)
#plots_e["al_err"].set_ylim(top=1.4)
#plots_e["ti_err"].set_ylim(top=2.5)
#
## Seperates each subplot such that overlap of subplots and axes titles won't be present
#plt.tight_layout(w_pad=-3.5, h_pad=-1)
#plt.show()
#
##
## The following is to validate the above model.
##
#
#
#def cross_validation(n: int, k: int, data: pd.DataFrame):
#    """A cross-validation algorithm to validate a linear model
#
#    Parameters
#    ----------
#    n : int
#        The number of times to run the algorithm
#    k : int
#        The number of samples we obtain from the data
#    data : pd.DataFrame
#        The data to validate with
#
#    Returns
#    -------
#    errors_df : pd.DataFrame
#        A DataFrame containing the errors calculated from the model
#    """
#    dataframe_test = DataframeWriter(data)
#    # Loops n * (k - 1) times as if we split data by 3 we can do 2 cross-validations for that data
#    loop_range = n * (k - 1)
#    # Create empty array to fill with error for every loop
#    logged_std: np.ndarray = np.zeros([3, loop_range])
#    logged_r_squared: np.ndarray = np.zeros([3, loop_range])
#    logged_mape: np.ndarray = np.zeros([3, loop_range])
#
#    for i in range(0, loop_range, k - 1):
#        # Generates stratified samples
#        samples: list[pd.DataFrame] = dataframe_test.data_shuffler(k)
#        # Sets the first sample to be our control which the model is not trained on
#        control_sample: pd.DataFrame = samples[0].dropna()
#        properties_array: np.ndarray = np.array(
#            control_sample["combined_properties"].values.tolist()
#        )
#
#        # Trains sample for each non-control sample
#        for j in range(0, k - 1):
#            A_learned = a_calc(samples[j + 1])
#
#            predicted_array = np.array(
#                control_sample["combined_compositions"]
#                .apply(lambda prop: np.matmul(A_learned, prop))
#                .values.tolist()
#            )
#
#            r_squared = r2_score(
#                properties_array,
#                predicted_array,
#                multioutput="raw_values",
#            )
#
#            # Stores each value of error calculated
#            logged_r_squared[:, i + j] = r_squared
#            logged_std[:, i + j] = calculate_std(properties_array, predicted_array)
#            logged_mape[:, i + j] = calculate_mape(properties_array, predicted_array)
#
#    # Stores data within a DataFrame
#    errors_df: pd.DataFrame = pd.DataFrame()
#    errors_df["yield_strength_R^2"] = logged_r_squared[0]
#    errors_df["tensile_strength_R^2"] = logged_r_squared[1]
#    errors_df["elongation_R^2"] = logged_r_squared[2]
#    errors_df["yield_strength_std"] = logged_std[0]
#    errors_df["tensile_strength_std"] = logged_std[1]
#    errors_df["elongation_std"] = logged_std[2]
#    errors_df["yield_strength_mape"] = logged_mape[0]
#    errors_df["tensile_strength_mape"] = logged_mape[1]
#    errors_df["elongation_mape"] = logged_mape[2]
#
#    return errors_df
#
#
## Runs cross-validation
#errors_df = cross_validation(1000, 3, test_alloy_properties)
#
## Generates distributions for each error
#figure, axis = plt.subplots(3, 3)
#
#sns.kdeplot(data=errors_df, x="yield_strength_R^2", clip=(0.0, 1.0), ax=axis[0, 0])
#sns.kdeplot(data=errors_df, x="tensile_strength_R^2", clip=(0.0, 1.0), ax=axis[0, 1])
#sns.kdeplot(data=errors_df, x="elongation_R^2", clip=(0.0, 1.0), ax=axis[0, 2])
#sns.kdeplot(data=errors_df, x="yield_strength_std", clip=(0.0, 500.0), ax=axis[1, 0])
#sns.kdeplot(data=errors_df, x="tensile_strength_std", clip=(0.0, 500.0), ax=axis[1, 1])
#sns.kdeplot(data=errors_df, x="elongation_std", clip=(0.0, 12.5), ax=axis[1, 2])
#sns.kdeplot(data=errors_df, x="yield_strength_mape", clip=(0.0, 100.0), ax=axis[2, 0])
#sns.kdeplot(data=errors_df, x="tensile_strength_mape", clip=(0.0, 100.0), ax=axis[2, 1])
#sns.kdeplot(data=errors_df, x="elongation_mape", clip=(0.0, 100.0), ax=axis[2, 2])
#
#plt.show()

#
# The following is an algorithm to find an optimal minimum composition of Co and Ni for best mechanical properties
# and Ni content for best mechanical properties
#

bounds: np.ndarray = np.array(
    [
        [
            refined_alloy_properties[element].min(),
            refined_alloy_properties[element].max(),
        ]
        for element in ELEMENTS
    ]
)

bounds_tuple_list = list(map(tuple, bounds.reshape((14, 2))))

initial_ranges: np.ndarray = np.array(
    [
        [
            0.75 * bounds[i, 0] + 0.25 * bounds[i, 1],
            0.75 * bounds[i, 1] + 0.25 * bounds[i, 0],
        ]
        for i in range(0, 14)
    ]
)

x_start = np.random.uniform(
    low=initial_ranges[:, 0].transpose(),
    high=initial_ranges[:, 1].transpose(),
    size=(1, 14),
).transpose()

x_start = x_start * 100 / np.sum(x_start)

w1 = 0.0025
w2 = 1
analytical_weights_intermediate = np.mean(
    reduced_alloy_properties["combined_predicted"]
)
analytical_weights = (
    np.sum(analytical_weights_intermediate) / analytical_weights_intermediate
)
alpha = analytical_weights[0]
beta = analytical_weights[1]
gamma = analytical_weights[2]

y_current = np.matmul(A_learned, x_start)

function_to_min = x_start[5] + x_start[10]
function = alpha * y_current[0] + beta * y_current[1] + gamma * y_current[2]

x_current = x_start

print(sum(x_current))

def total_function(x: np.ndarray):
   x_current[5] = x[0]
   x_current[10] = x[1]
   ratio = (100 - sum(x)) / (sum(x_current) - sum(x))
   x_current[0:5] = [np.round(x_current[i] * ratio, 2) for i in range(0, 5)]
   x_current[6:10] = [np.round(x_current[i] * ratio, 2) for i in range(5, 9)]
   x_current[11:14] = [np.round(x_current[i] * ratio, 2) for i in range(9, 12)]
   function_to_min = sum(x)
   y = np.matmul(A_learned, x_current)
   function_to_max = alpha * y[0] + beta * y[1] + gamma * y[2]
   return w1 * function_to_max - w2 * function_to_min

for _ in range(0, 1000):
    x_start = np.random.uniform(
        low=initial_ranges[:, 0].transpose(),
        high=initial_ranges[:, 1].transpose(),
        size=(1, 14),
    ).transpose()

    x_start = x_start * 100 / np.sum(x_start)
    ans = optimize.minimize(total_function, np.array([x_start[5], x_start[10]]).reshape(2), bounds=[bounds_tuple_list[5], bounds_tuple_list[10]])

    if sum(x_current) > 105:
        print(sum(x_current))
        print(ans.x)