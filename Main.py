#Main 

import numpy as np
import pandas as pd
import sklearn.linear_model

ELEMENT_WEIGHT: dict[str, float] = {
    'Fe': 55.845,
    'C': 12.011,
    'Mn': 54.938,
    'Si': 28.086,
    'Cr': 51.996,
    'Ni': 58.693,
    'Mo': 95.94,
    'V':50.942,
    'N':14.007,
    'Nb': 92.906,
    'Co': 58.933,
    'W':183.84,
    'Al': 26.982,
    'Ti': 47.867
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
            fe_calc.append(weight_dict.get('Fe', 0))
            c_calc.append(weight_dict.get('C', 0))
            mn_calc.append(weight_dict.get('Mn', 0))
            si_calc.append(weight_dict.get('Si', 0))
            cr_calc.append(weight_dict.get('Cr', 0))
            ni_calc.append(weight_dict.get('Ni', 0))
            mo_calc.append(weight_dict.get('Mo', 0))
            v_calc.append(weight_dict.get('V', 0))
            n_calc.append(weight_dict.get('N', 0))
            nb_calc.append(weight_dict.get('Nb', 0))
            co_calc.append(weight_dict.get('Co', 0))
            w_calc.append(weight_dict.get('W', 0))
            al_calc.append(weight_dict.get('Al', 0))
            ti_calc.append(weight_dict.get('Ti', 0))

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
    composition_vectors = np.zeros([len(data.index), 14])
    properties_vectors = np.zeros([len(data.index), 3])

    for i in range(0, len(data.index)):
        if np.isnan(alloy_properties['tensile strength'][i]) or np.isnan(alloy_properties['yield strength'][i]) or np.isnan(alloy_properties['elongation'][i]):
            continue

        composition_vectors[i] = np.array([
            data.fe_calc[i],
            data.c.combine_first(data.c_calc)[i],
            data.mn.combine_first(data.mn_calc)[i],
            data.si.combine_first(data.si_calc)[i],
            data.cr.combine_first(data.cr_calc)[i],
            data.ni.combine_first(data.ni_calc)[i],
            data.mo.combine_first(data.mo_calc)[i],
            data.v.combine_first(data.v_calc)[i],
            data.n.combine_first(data.n_calc)[i],
            data.nb.combine_first(data.nb_calc)[i],
            data.co.combine_first(data.co_calc)[i],
            data.w.combine_first(data.w_calc)[i],
            data.al.combine_first(data.al_calc)[i],
            data.ti.combine_first(data.ti_calc)[i]
        ])
        
        properties_vectors[i] = np.array([
        data.loc[i]["yield strength"],
        data.loc[i]["tensile strength"],
        data.loc[i]["elongation"]
        ])
        
    model = sklearn.linear_model.LinearRegression(fit_intercept = False)
    model.fit(composition_vectors, properties_vectors)

    A = model.coef_

    return A
        
alloy_properties = pd.read_csv(r"C:\Users\sambi\Programming\alloy-properties-ml\database_steel_properties.csv", skiprows=1)

dataframe_writer = DataframeWriter(alloy_properties)

alloy_properties = dataframe_writer.data_fill()

A_learned = a_calc(alloy_properties)

y_sum_1 = 0
y_sum_2 = 0
y_sum_3 = 0
num_check = 0

for k in range(0, len(alloy_properties.index)):
    if np.isnan(alloy_properties['tensile strength'][k]) or np.isnan(alloy_properties['yield strength'][k]) or np.isnan(alloy_properties['elongation'][k]):
        continue

    num_check +=1
    composition_vector = np.array([
        [alloy_properties.fe_calc[k]],
        [alloy_properties.c.combine_first(alloy_properties.c_calc)[k]],
        [alloy_properties.mn.combine_first(alloy_properties.mn_calc)[k]],
        [alloy_properties.si.combine_first(alloy_properties.si_calc)[k]],
        [alloy_properties.cr.combine_first(alloy_properties.cr_calc)[k]],
        [alloy_properties.ni.combine_first(alloy_properties.ni_calc)[k]],
        [alloy_properties.mo.combine_first(alloy_properties.mo_calc)[k]],
        [alloy_properties.v.combine_first(alloy_properties.v_calc)[k]],
        [alloy_properties.n.combine_first(alloy_properties.n_calc)[k]],
        [alloy_properties.nb.combine_first(alloy_properties.nb_calc)[k]],
        [alloy_properties.co.combine_first(alloy_properties.co_calc)[k]],
        [alloy_properties.w.combine_first(alloy_properties.w_calc)[k]],
        [alloy_properties.al.combine_first(alloy_properties.al_calc)[k]],
        [alloy_properties.ti.combine_first(alloy_properties.ti_calc)[k]]
    ])

    properties_vector = np.array([
        alloy_properties["yield strength"][k],
        alloy_properties["tensile strength"][k],
        alloy_properties["elongation"][k]
    ])

    y_calc = np.matmul(A_learned, composition_vector)
    y_sum_1 += abs(properties_vector[0] - y_calc[0])
    y_sum_2 += abs(properties_vector[1] - y_calc[1])
    y_sum_3 += abs(properties_vector[2] - y_calc[2])

average_y_deviation = np.array([y_sum_1, y_sum_2, y_sum_3]) / num_check

print(average_y_deviation)