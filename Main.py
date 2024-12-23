#Main 

import numpy as np
import pandas as pd

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
    def __init__(self):
        pass

    def atom_to_weight_percent(formula: str):
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

    def data_fill(self, data: pd.DataFrame):
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

        for i in range(0, len(data.index)):
            weight_dict: dict = self.atom_to_weight_percent(data.iat[i, 0])
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

        data['fe_calc'] = fe_calc
        data['c_calc'] = c_calc
        data['mn_calc'] = mn_calc
        data['si_calc'] = si_calc
        data['cr_calc'] = cr_calc
        data['ni_calc'] = ni_calc
        data['mo_calc'] = mo_calc
        data['v_calc'] = v_calc
        data['n_calc'] = n_calc
        data['nb_calc'] = nb_calc
        data['co_calc'] = co_calc
        data['w_calc'] = w_calc
        data['al_calc'] = al_calc
        data['ti_calc'] = ti_calc

        return data

def grad_descent(data: pd.DataFrame):
    data['a'] = np.zeros(len(data.index))
    for i in range(0, len(data.index)):
        learning_rate = 0.1
        composition_vector = np.array([
            data.fe[i],
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

        properties_vector = np.array([
            data.loc[i]["yield strength"]
            data.loc[i]["tensile strength"]
            data.loc[i]["elongation"]
        ])

        a = np.zeros([3])
        for j in range(0, 3):
            a_n = np.random.rand(14)
            continue_iterations = True
            while continue_iterations:
                a_n_1 = a_n - learning_rate * (properties_vector[j] - np.dot(a_n, composition_vector))
                if abs(a_n_1 - a_n) < 10:
                    continue_iterations = False
                    a[j] = a_n_1
        
        data['a'][i] = a