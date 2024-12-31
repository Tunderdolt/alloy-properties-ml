#Main 

from typing import Literal, get_args
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.metrics import r2_score

Element = Literal["fe", "c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]
Property = Literal["yield_strength", "tensile_strength", "elongation"]

ELEMENTS: list[Element] = list(get_args(Element))
PROPERTIES: list[Property] = list(get_args(Property))

ELEMENT_WEIGHT: dict[str, float] = {
    'fe': 55.845,
    'c': 12.011,
    'mn': 54.938,
    'si': 28.086,
    'cr': 51.996,
    'ni': 58.693,
    'mo': 95.94,
    'v': 50.942,
    'n': 14.007,
    'nb': 92.906,
    'co': 58.933,
    'w': 183.84,
    'al': 26.982,
    'ti': 47.867
}

class DataframeWriter:
    def __init__(self, data):
        self.data = data

    def atom_to_weight_percent(self, formula: str):
        atom_list = list(formula)

        flip = False
        joined_list = []
        k = 0

        for i in range(1, len(atom_list)):
            if not flip:
                k += 1
                if not atom_list[i].isalpha():
                    flip = True

                    if k == 1:
                        joined_list.append(atom_list[i-1])

                    elif k == 2:
                        joined_list.append(atom_list[i-2] + atom_list[i-1])

                    k = 0

            elif flip:
                k += 1
                if atom_list[i].isalpha():
                    flip = False

                    intermediate = ''
                    for j in range(i - k, i):
                        intermediate += atom_list[j]
                    joined_list.append(intermediate)

                    k = 0

                elif i == len(atom_list) - 1:
                    intermediate = ''
                    for j in range(i - k, i + 1):
                        intermediate += atom_list[j]
                    joined_list.append(intermediate)

        formula_tuple_list: list[tuple[str, float]] = []

        for i in range(0, len(joined_list), 2):
            formula_tuple_list.append(tuple((joined_list[i], float(joined_list[i+1]))))

        formula_dict = dict(formula_tuple_list)
        weight_dict_intermediate: dict[str, float] = {}

        for key in formula_dict:
            weight_dict_intermediate[key] = formula_dict[key] * ELEMENT_WEIGHT[key]


        total_weight = sum(weight_dict_intermediate.values())

        weight_dict = {}
        for key in weight_dict_intermediate:
            weight_dict[key] = round(100 * weight_dict_intermediate[key] / total_weight, 2)

        return weight_dict

    def data_fill(self):
        fe_calc = []
        c_calc = []
        mn_calc = []
        si_calc = []
        cr_calc = []
        ni_calc = []
        mo_calc = []
        v_calc = []
        n_calc = []
        nb_calc = []
        co_calc = []
        w_calc = []
        al_calc = []
        ti_calc = []

        for i in range(0, len(self.data.index)):
            weight_dict: dict = self.atom_to_weight_percent(self.data['formula'][i])
            fe_calc.append(weight_dict.get('fe', 0))
            c_calc.append(weight_dict.get('c', 0))
            mn_calc.append(weight_dict.get('mn', 0))
            si_calc.append(weight_dict.get('si', 0))
            cr_calc.append(weight_dict.get('cr', 0))
            ni_calc.append(weight_dict.get('ni', 0))
            mo_calc.append(weight_dict.get('mo', 0))
            v_calc.append(weight_dict.get('v', 0))
            n_calc.append(weight_dict.get('n', 0))
            nb_calc.append(weight_dict.get('nb', 0))
            co_calc.append(weight_dict.get('co', 0))
            w_calc.append(weight_dict.get('w', 0))
            al_calc.append(weight_dict.get('al', 0))
            ti_calc.append(weight_dict.get('ti', 0))

        self.data['fe_calc'] = fe_calc
        self.data['c_calc'] = c_calc
        self.data['mn_calc'] = mn_calc
        self.data['si_calc'] = si_calc
        self.data['cr_calc'] = cr_calc
        self.data['ni_calc'] = ni_calc
        self.data['mo_calc'] = mo_calc
        self.data['v_calc'] = v_calc
        self.data['n_calc'] = n_calc
        self.data['nb_calc'] = nb_calc
        self.data['co_calc'] = co_calc
        self.data['w_calc'] = w_calc
        self.data['al_calc'] = al_calc
        self.data['ti_calc'] = ti_calc

        return self.data

def a_calc(data: pd.DataFrame):
    data = data.dropna()
    composition_vectors = data[[element for element in ELEMENTS]].to_numpy()
    properties_vectors = data[[properties for properties in PROPERTIES]].to_numpy()

    model = sklearn.linear_model.LinearRegression(fit_intercept = False)
    model.fit(composition_vectors, properties_vectors)

    A = model.coef_

    return A
        
alloy_properties = pd.read_csv(r"C:\Users\sambi\Programming\alloy-properties-ml\database_steel_properties.csv", skiprows=1)
alloy_properties['formula'] = alloy_properties['formula'].apply(lambda value: value.lower())

dataframe_writer = DataframeWriter(alloy_properties)

alloy_properties = dataframe_writer.data_fill()

refined_alloy_properties = pd.DataFrame(
    {
        element: alloy_properties[element].combine_first(alloy_properties[f"{element}_calc"])
        for element in ELEMENTS if element != "fe"
    }
)
refined_alloy_properties["fe"] = alloy_properties["fe_calc"]
refined_alloy_properties["combined_compositions"] = refined_alloy_properties[[element for element in ELEMENTS]].values.tolist()
refined_alloy_properties["yield_strength"] = alloy_properties["yield strength"]
refined_alloy_properties["tensile_strength"] = alloy_properties["tensile strength"]
refined_alloy_properties["elongation"] = alloy_properties["elongation"]
refined_alloy_properties["combined_properties"] = refined_alloy_properties[[prop for prop in PROPERTIES]].values.tolist()
refined_alloy_properties["combined_properties"] = refined_alloy_properties["combined_properties"].apply(lambda prop: np.array(prop))

A_learned = a_calc(refined_alloy_properties)

composition_vectors = refined_alloy_properties[[element for element in ELEMENT_WEIGHT.keys()]].to_numpy()

refined_alloy_properties["elongation_predicted"] = refined_alloy_properties["combined_compositions"].apply(lambda prop: np.matmul(A_learned, prop)[2])
refined_alloy_properties["combined_predicted"] = refined_alloy_properties["combined_compositions"].apply(lambda prop: np.matmul(A_learned, prop))

reduced_alloy_properties = refined_alloy_properties.dropna()

properties_array = np.array(reduced_alloy_properties["combined_properties"].values.tolist())
predicted_array = np.array(reduced_alloy_properties["combined_predicted"].values.tolist())

r_squared = r2_score(
    properties_array,
    predicted_array,
    multioutput="raw_values",
)

print(r_squared)

difference_array = properties_array - predicted_array

std_yield_strength = np.sqrt(np.sum(np.square(difference_array), axis=0)[0] / difference_array.shape[0])
std_tensile_strength = np.sqrt(np.sum(np.square(difference_array), axis=0)[1] / difference_array.shape[0])
std_elongation = np.sqrt(np.sum(np.square(difference_array), axis=0)[2] / difference_array.shape[0])

print(std_yield_strength)
print(std_tensile_strength)
print(std_elongation)

print(refined_alloy_properties)

ElongationCategory = Literal["weak", "medium", "strong", "NaN"]

def categorise_elongation(elongation: float) -> ElongationCategory:
    if np.isnan(elongation):
        return np.nan
    if elongation < 5:
        return "weak"
    if elongation > 10:
        return "strong"
    return "medium"

refined_alloy_properties["elongation_catagorised_true"] = refined_alloy_properties["elongation"].apply(categorise_elongation)

refined_alloy_properties["elongation_catagorised_predicted"] = refined_alloy_properties["elongation_predicted"].apply(categorise_elongation)

print(refined_alloy_properties)

count_have_data = 0
count_not_equal = 0

for i in range(0, len(refined_alloy_properties.index)):
    if refined_alloy_properties.loc[i, "elongation_catagorised_true"] == "NaN":
        continue
    count_have_data += 1
    if refined_alloy_properties.loc[i, "elongation_catagorised_true"] != refined_alloy_properties.loc[i, "elongation_catagorised_predicted"]:
        count_not_equal +=1

print(count_not_equal / count_have_data)
