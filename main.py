#main 

from typing import Literal, get_args
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None

Element = Literal["fe", "c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]
Property = Literal["yield_strength", "tensile_strength", "elongation"]
ElongationCategory = Literal["weak", "medium", "strong", "NaN"]

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
    def __init__(self, data: pd.DataFrame):
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
    
    @staticmethod
    def data_shuffler(data: pd.DataFrame, n: int):
        sorted_data = data.sort_values(["fe"], ignore_index=True)
        length_of_segment = int(round(len(data.index) / n))
        shuffled_fe_dict: dict[str, pd.DataFrame] = {}

        for i in range(0, n-1):
            shuffled_fe_dict[f"shuffled_fe_data{i+1}"] = sorted_data.iloc[i*length_of_segment:(i+1)*length_of_segment].sample(frac=1).reset_index(drop=True)

        shuffled_fe_dict[f"shuffled_fe_data{n}"] = sorted_data.iloc[(n-1)*length_of_segment:len(data.index)].sample(frac=1).reset_index(drop=True)

        fe_sample_data: list[pd.DataFrame] = []
        partial_sample_dict: dict[str, pd.DataFrame] = {}

        for i in range(1, n):
            sample_df = pd.DataFrame([])
            for j in range(1, n+1):
                partial_sample_dict[f"partial_sample{j}"] = shuffled_fe_dict[f"shuffled_fe_data{j}"].iloc[0:int(round(length_of_segment / n))]
                sample_df = pd.concat([sample_df, partial_sample_dict[f"partial_sample{j}"]])
                shuffled_fe_dict[f"shuffled_fe_data{j}"] = shuffled_fe_dict[f"shuffled_fe_data{j}"].reset_index(drop=True).drop(np.arange(0, int(round(length_of_segment / n))))
            
            fe_sample_data.append(sample_df.reset_index(drop=True))
        
        sample_df = pd.DataFrame([])

        for i in range(1, n+1):
            sample_df = pd.concat([sample_df, shuffled_fe_dict[f"shuffled_fe_data{i}"]])
        
        fe_sample_data.append(sample_df.reset_index(drop=True))

        return fe_sample_data


def a_calc(data: pd.DataFrame):
    data = data.dropna()
    composition_vectors = data[[element for element in ELEMENTS]].to_numpy()
    properties_vectors = data[[properties for properties in PROPERTIES]].to_numpy()

    model = sklearn.linear_model.LinearRegression(fit_intercept = False)
    model.fit(composition_vectors, properties_vectors)

    A = model.coef_

    return A

def calculate_mape(actual, predicted) -> float:  
    if not all([isinstance(actual, np.ndarray), 
                isinstance(predicted, np.ndarray)]): 
        actual, predicted = np.array(actual),  
        np.array(predicted) 

    return round(np.mean(np.abs(( 
      actual - predicted) / actual)) * 100, 2)

def categorise_elongation(elongation: float) -> ElongationCategory:
    if np.isnan(elongation):
        return np.nan
    if elongation < 5:
        return "weak"
    if elongation > 10:
        return "strong"
    return "medium"
        
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

test_alloy_properties = refined_alloy_properties

A_learned = a_calc(refined_alloy_properties)

composition_vectors = refined_alloy_properties[[element for element in ELEMENT_WEIGHT.keys()]].to_numpy()

refined_alloy_properties["elongation_predicted"] = refined_alloy_properties["combined_compositions"].apply(lambda prop: np.matmul(A_learned, prop)[2])
refined_alloy_properties["combined_predicted"] = refined_alloy_properties["combined_compositions"].apply(lambda prop: np.matmul(A_learned, prop))

reduced_alloy_properties = refined_alloy_properties.dropna()

properties_array = np.array(reduced_alloy_properties["combined_properties"].values.tolist())
predicted_array = np.array(reduced_alloy_properties["combined_predicted"].values.tolist())

r_squared = list(r2_score(
    properties_array,
    predicted_array,
    multioutput="raw_values",
))

difference_array = properties_array - predicted_array

std_yield_strength = np.sqrt(np.sum(np.square(difference_array), axis=0)[0] / difference_array.shape[0])
std_tensile_strength = np.sqrt(np.sum(np.square(difference_array), axis=0)[1] / difference_array.shape[0])
std_elongation = np.sqrt(np.sum(np.square(difference_array), axis=0)[2] / difference_array.shape[0])

mape_ys = calculate_mape(properties_array[:,0], predicted_array[:,0])
mape_ts = calculate_mape(properties_array[:,1], predicted_array[:,1])
mape_e = calculate_mape(properties_array[:,2], predicted_array[:,2])

refined_alloy_properties["elongation_catagorised_true"] = refined_alloy_properties["elongation"].apply(categorise_elongation)
refined_alloy_properties["elongation_catagorised_predicted"] = refined_alloy_properties["elongation_predicted"].apply(categorise_elongation)

count_have_data = 0 
count_not_equal = 0

for i in range(0, len(refined_alloy_properties.index)):
    if refined_alloy_properties.loc[i, "elongation_catagorised_true"] == "NaN":
        continue
    count_have_data += 1
    if refined_alloy_properties.loc[i, "elongation_catagorised_true"] != refined_alloy_properties.loc[i, "elongation_catagorised_predicted"]:
        count_not_equal +=1

print(f"The R^2 for the entire dataset is {r_squared}")
print(f"The std for the entire dataset is {[std_yield_strength, std_tensile_strength, std_elongation]}")
print(f"The MAPE for the entire dataset is {[mape_ys, mape_ts, mape_e]}")
print(f"The percentage of incorrect assignments of elongation is {count_not_equal / count_have_data}")

#
#The following is to see if there is a correlation between error and and element composition
#

refined_alloy_properties["percent_error_ys"] = abs((refined_alloy_properties["combined_properties"].apply(lambda prop: prop[0]) - refined_alloy_properties["combined_predicted"]).apply(lambda prop: prop[0]) / refined_alloy_properties["combined_properties"].apply(lambda prop: prop[0]))
refined_alloy_properties["percent_error_ts"] = abs((refined_alloy_properties["combined_properties"].apply(lambda prop: prop[1]) - refined_alloy_properties["combined_predicted"]).apply(lambda prop: prop[1]) / refined_alloy_properties["combined_properties"].apply(lambda prop: prop[1]))
refined_alloy_properties["percent_error_e"] = abs((refined_alloy_properties["combined_properties"].apply(lambda prop: prop[2]) - refined_alloy_properties["combined_predicted"].apply(lambda prop: prop[2])) / refined_alloy_properties["combined_properties"].apply(lambda prop: prop[2]))
refined_alloy_properties = refined_alloy_properties.dropna()

figure, axis = plt.subplots(2, 7)

fe_e = sns.kdeplot(data=refined_alloy_properties, y="fe", x="percent_error_ys", ax=axis[0,0], fill=True, cmap="rocket_r")
fe_e.set_ylim([60, 85])
fe_e.set_xlim(left=0)

c_e = sns.kdeplot(data=refined_alloy_properties, y="c", x="percent_error_ys", ax=axis[0,1], fill=True, cmap="rocket_r")
c_e.set_ylim([0,0.45])
c_e.set_xlim(left=0)

mn_e = sns.kdeplot(data=refined_alloy_properties, y="mn", x="percent_error_ys", ax=axis[0,2], fill=True, cmap="rocket_r")
mn_e.set_ylim([0,1])
mn_e.set_xlim(left=0)

si_e = sns.kdeplot(data=refined_alloy_properties, y="si", x="percent_error_ys", ax=axis[0,3], fill=True, cmap="rocket_r")
si_e.set_ylim([0,2.5])
si_e.set_xlim(left=0)

cr_e = sns.kdeplot(data=refined_alloy_properties, y="cr", x="percent_error_ys", ax=axis[0,4], fill=True, cmap="rocket_r")
cr_e.set_ylim([0,22.5])
cr_e.set_xlim(left=0)

ni_e = sns.kdeplot(data=refined_alloy_properties, y="ni", x="percent_error_ys", ax=axis[0,5], fill=True, cmap="rocket_r")
ni_e.set_ylim([0,25])
ni_e.set_xlim(left=0)

mo_e = sns.kdeplot(data=refined_alloy_properties, y="mo", x="percent_error_ys", ax=axis[0,6], fill=True, cmap="rocket_r")
mo_e.set_ylim([0,8])
mo_e.set_xlim(left=0)

v_e = sns.kdeplot(data=refined_alloy_properties, y="v", x="percent_error_ys", ax=axis[1,0], fill=True, cmap="rocket_r")
v_e.set_ylim([0,1.4])
v_e.set_xlim(left=0)

n_e = sns.kdeplot(data=refined_alloy_properties, y="n", x="percent_error_ys", ax=axis[1,1], fill=True, cmap="rocket_r")
n_e.set_ylim([0,0.055])
n_e.set_xlim(left=0)

nb_e = sns.kdeplot(data=refined_alloy_properties, y="nb", x="percent_error_ys", ax=axis[1,2], fill=True, cmap="rocket_r")
nb_e.set_ylim([0,0.25])
nb_e.set_xlim(left=0)

co_e = sns.kdeplot(data=refined_alloy_properties, y="co", x="percent_error_ys", ax=axis[1,3], fill=True, cmap="rocket_r")
co_e.set_ylim([0,22.5])
co_e.set_xlim(left=0)

w_e = sns.kdeplot(data=refined_alloy_properties, y="w", x="percent_error_ys", ax=axis[1,4], fill=True, cmap="rocket_r")
w_e.set_ylim([0,2.5])
w_e.set_xlim(left=0)

al_e = sns.kdeplot(data=refined_alloy_properties, y="al", x="percent_error_ys", ax=axis[1,5], fill=True, cmap="rocket_r")
al_e.set_ylim([0,1.4])
al_e.set_xlim(left=0)

ti_e = sns.kdeplot(data=refined_alloy_properties, y="ti", x="percent_error_ys", ax=axis[1,6], fill=True, cmap="rocket_r")
ti_e.set_ylim([0,2.75])
ti_e.set_xlim(left=0)

plt.tight_layout(w_pad=-2.5, h_pad=-1)
plt.show()

figure, axis = plt.subplots(2, 7)

fe_e = sns.kdeplot(data=refined_alloy_properties, y="fe", x="percent_error_ts", ax=axis[0,0], fill=True, cmap="copper_r")
fe_e.set_ylim([60, 85])
fe_e.set_xlim(left=0)

c_e = sns.kdeplot(data=refined_alloy_properties, y="c", x="percent_error_ts", ax=axis[0,1], fill=True, cmap="copper_r")
c_e.set_ylim([0,0.5])
c_e.set_xlim(left=0)

mn_e = sns.kdeplot(data=refined_alloy_properties, y="mn", x="percent_error_ts", ax=axis[0,2], fill=True, cmap="copper_r")
mn_e.set_ylim([0,1.1])
mn_e.set_xlim(left=0)

si_e = sns.kdeplot(data=refined_alloy_properties, y="si", x="percent_error_ts", ax=axis[0,3], fill=True, cmap="copper_r")
si_e.set_ylim([0,2.5])
si_e.set_xlim(left=0)

cr_e = sns.kdeplot(data=refined_alloy_properties, y="cr", x="percent_error_ts", ax=axis[0,4], fill=True, cmap="copper_r")
cr_e.set_ylim([0,22.5])
cr_e.set_xlim(left=0)

ni_e = sns.kdeplot(data=refined_alloy_properties, y="ni", x="percent_error_ts", ax=axis[0,5], fill=True, cmap="copper_r")
ni_e.set_ylim([0,25])
ni_e.set_xlim(left=0)

mo_e = sns.kdeplot(data=refined_alloy_properties, y="mo", x="percent_error_ts", ax=axis[0,6], fill=True, cmap="copper_r")
mo_e.set_ylim([0,8])
mo_e.set_xlim(left=0)

v_e = sns.kdeplot(data=refined_alloy_properties, y="v", x="percent_error_ts", ax=axis[1,0], fill=True, cmap="copper_r")
v_e.set_ylim([0,1.4])
v_e.set_xlim(left=0)

n_e = sns.kdeplot(data=refined_alloy_properties, y="n", x="percent_error_ts", ax=axis[1,1], fill=True, cmap="copper_r")
n_e.set_ylim([0,0.055])
n_e.set_xlim(left=0)

nb_e = sns.kdeplot(data=refined_alloy_properties, y="nb", x="percent_error_ts", ax=axis[1,2], fill=True, cmap="copper_r")
nb_e.set_ylim([0,0.25])
nb_e.set_xlim(left=0)

co_e = sns.kdeplot(data=refined_alloy_properties, y="co", x="percent_error_ts", ax=axis[1,3], fill=True, cmap="copper_r")
co_e.set_ylim([0,22.5])
co_e.set_xlim(left=0)

w_e = sns.kdeplot(data=refined_alloy_properties, y="w", x="percent_error_ts", ax=axis[1,4], fill=True, cmap="copper_r")
w_e.set_ylim([0,2.75])
w_e.set_xlim(left=0)

al_e = sns.kdeplot(data=refined_alloy_properties, y="al", x="percent_error_ts", ax=axis[1,5], fill=True, cmap="copper_r")
al_e.set_ylim([0,1.4])
al_e.set_xlim(left=0)

ti_e = sns.kdeplot(data=refined_alloy_properties, y="ti", x="percent_error_ts", ax=axis[1,6], fill=True, cmap="copper_r")
ti_e.set_ylim([0,2.75])
ti_e.set_xlim(left=0)

plt.tight_layout(w_pad=-2.5, h_pad=-1)
plt.show()

figure, axis = plt.subplots(2, 7)

fe_e = sns.kdeplot(data=refined_alloy_properties, y="fe", x="percent_error_e", ax=axis[0,0], fill=True, cmap="cividis_r")
fe_e.set_ylim([60, 85])
fe_e.set_xlim([0,2.5])

c_e = sns.kdeplot(data=refined_alloy_properties, y="c", x="percent_error_e", ax=axis[0,1], fill=True, cmap="cividis_r")
c_e.set_ylim([0,0.5])
c_e.set_xlim([0,2.25])

mn_e = sns.kdeplot(data=refined_alloy_properties, y="mn", x="percent_error_e", ax=axis[0,2], fill=True, cmap="cividis_r")
mn_e.set_ylim([0,1.2])
mn_e.set_xlim([0,2.5])

si_e = sns.kdeplot(data=refined_alloy_properties, y="si", x="percent_error_e", ax=axis[0,3], fill=True, cmap="cividis_r")
si_e.set_ylim([0,2.5])
si_e.set_xlim([0,2.5])

cr_e = sns.kdeplot(data=refined_alloy_properties, y="cr", x="percent_error_e", ax=axis[0,4], fill=True, cmap="cividis_r")
cr_e.set_ylim([0,22.5])
cr_e.set_xlim([0,2.25])

ni_e = sns.kdeplot(data=refined_alloy_properties, y="ni", x="percent_error_e", ax=axis[0,5], fill=True, cmap="cividis_r")
ni_e.set_ylim([0,25])
ni_e.set_xlim([0,2.5])

mo_e = sns.kdeplot(data=refined_alloy_properties, y="mo", x="percent_error_e", ax=axis[0,6], fill=True, cmap="cividis_r")
mo_e.set_ylim([0,8])
mo_e.set_xlim([0,3])

v_e = sns.kdeplot(data=refined_alloy_properties, y="v", x="percent_error_e", ax=axis[1,0], fill=True, cmap="cividis_r")
v_e.set_ylim([0,2.25])
v_e.set_xlim([0,2.5])

n_e = sns.kdeplot(data=refined_alloy_properties, y="n", x="percent_error_e", ax=axis[1,1], fill=True, cmap="cividis_r")
n_e.set_ylim([0,0.06])
n_e.set_xlim([0,2.5])

nb_e = sns.kdeplot(data=refined_alloy_properties, y="nb", x="percent_error_e", ax=axis[1,2], fill=True, cmap="cividis_r")
nb_e.set_ylim([0,0.36])
nb_e.set_xlim([0,2.5])

co_e = sns.kdeplot(data=refined_alloy_properties, y="co", x="percent_error_e", ax=axis[1,3], fill=True, cmap="cividis_r")
co_e.set_ylim([0,22.5])
co_e.set_xlim([0,2.5])

w_e = sns.kdeplot(data=refined_alloy_properties, y="w", x="percent_error_e", ax=axis[1,4], fill=True, cmap="cividis_r")
w_e.set_ylim([0,2.25])
w_e.set_xlim([0,2.5])

al_e = sns.kdeplot(data=refined_alloy_properties, y="al", x="percent_error_e", ax=axis[1,5], fill=True, cmap="cividis_r")
al_e.set_ylim([0,1.4])
al_e.set_xlim([0,2.5])

ti_e = sns.kdeplot(data=refined_alloy_properties, y="ti", x="percent_error_e", ax=axis[1,6], fill=True, cmap="cividis_r")
ti_e.set_ylim([0,2.5])
ti_e.set_xlim([0,2.5])

plt.tight_layout(w_pad=-3.5, h_pad=-1)
plt.show()

#
#The following is to validate the above model.
#

def cross_validation(n, k, data: pd.DataFrame):
    logged_std_ys = np.zeros(n)
    logged_std_ts = np.zeros(n)
    logged_std_e = np.zeros(n)
    logged_r_squared_ys = np.zeros(n)
    logged_r_squared_ts = np.zeros(n)
    logged_r_squared_e = np.zeros(n)
    logged_mape_ys = np.zeros(n)
    logged_mape_ts = np.zeros(n)
    logged_mape_e = np.zeros(n)

    data = data.dropna()

    for i in range(0, n, k - 1):
        samples: list[pd.DataFrame] = DataframeWriter.data_shuffler(data, k)
        control_sample: pd.DataFrame = samples[0]
        properties_array = np.array(control_sample["combined_properties"].values.tolist())

        for j in range(0, k - 1):
            if i + j >= n:
                continue

            control_sample[f"combined_predicted_{j + 1}"] = (
                control_sample["combined_compositions"].apply(lambda prop: np.matmul(a_calc(samples[j + 1]), prop))
            )

            predicted_array = np.array(control_sample[f"combined_predicted_{j + 1}"].values.tolist())

            r_squared =  r2_score(
                properties_array,
                predicted_array,
                multioutput="raw_values",
            )

            logged_r_squared_ys[i + j] = r_squared[0]
            logged_r_squared_ts[i + j] = r_squared[1]
            logged_r_squared_e[i + j] = r_squared[2]

            difference_array = properties_array - predicted_array

            logged_std_ys[i + j] = np.sqrt(np.sum(np.square(difference_array[:,0]), axis=0) / difference_array.shape[0])
            logged_std_ts[i + j] = np.sqrt(np.sum(np.square(difference_array[:,1]), axis=0) / difference_array.shape[0])
            logged_std_e[i + j] = np.sqrt(np.sum(np.square(difference_array[:,2]), axis=0) / difference_array.shape[0])

            logged_mape_ys[i + j] = calculate_mape(properties_array[:,0], predicted_array[:,0])
            logged_mape_ts[i + j] = calculate_mape(properties_array[:,1], predicted_array[:,1])
            logged_mape_e[i + j] = calculate_mape(properties_array[:,2], predicted_array[:,2])

    errors_df = pd.DataFrame([])
    errors_df["yield_stress_R^2"] = logged_r_squared_ys
    errors_df["tensile_stress_R^2"] = logged_r_squared_ts
    errors_df["elongation_R^2"] = logged_r_squared_e
    errors_df["yield_stress_std"] = logged_std_ys
    errors_df["tensile_stress_std"] = logged_std_ts
    errors_df["elongation_std"] = logged_std_e
    errors_df["yield_stress_mape"] = logged_mape_ys
    errors_df["tensile_stress_mape"] = logged_mape_ts
    errors_df["elongation_mape"] = logged_mape_e

    return errors_df

errors_df = cross_validation(200, 3, test_alloy_properties)

figure, axis = plt.subplots(3, 3)

sns.kdeplot(data=errors_df, x="yield_stress_R^2", clip=(0.0,1.0), ax=axis[0, 0])
sns.kdeplot(data=errors_df, x="tensile_stress_R^2", clip=(0.0,1.0), ax=axis[0, 1])
sns.kdeplot(data=errors_df, x="elongation_R^2", clip=(0.0,1.0), ax=axis[0, 2])
sns.kdeplot(data=errors_df, x="yield_stress_std", clip=(0.0, 500.0), ax=axis[1, 0])
sns.kdeplot(data=errors_df, x="tensile_stress_std", clip=(0.0, 500.0), ax=axis[1, 1])
sns.kdeplot(data=errors_df, x="elongation_std", clip=(0.0, 12.5), ax=axis[1, 2])
sns.kdeplot(data=errors_df, x="yield_stress_mape", clip=(0.0,100.0), ax=axis[2, 0])
sns.kdeplot(data=errors_df, x="tensile_stress_mape", clip=(0.0,100.0), ax=axis[2, 1])
sns.kdeplot(data=errors_df, x="elongation_mape", clip=(0.0,100.0), ax=axis[2, 2])

plt.show()