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

class CleanDatabase:
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
        Fe_calc = []
        C_calc = []
        Mn_calc = []
        Si_calc = []
        Cr_calc = []
        Ni_calc = []
        Mo_calc = []
        V_calc = []
        N_calc = []
        Nb_calc = []
        Co_calc = []
        W_calc = []
        Al_calc = []
        Ti_calc = []

        for i in range(0, len(data.index)):
            weight_dict: dict = self.atom_to_weight_percent(data.iat[i, 0])
            Fe_calc.append(weight_dict.get('Fe', 0))
            C_calc.append(weight_dict.get('C', 0))
            Mn_calc.append(weight_dict.get('Mn', 0))
            Si_calc.append(weight_dict.get('Si', 0))
            Cr_calc.append(weight_dict.get('Cr', 0))
            Ni_calc.append(weight_dict.get('Ni', 0))
            Mo_calc.append(weight_dict.get('Mo', 0))
            V_calc.append(weight_dict.get('V', 0))
            N_calc.append(weight_dict.get('N', 0))
            Nb_calc.append(weight_dict.get('Nb', 0))
            Co_calc.append(weight_dict.get('Co', 0))
            W_calc.append(weight_dict.get('W', 0))
            Al_calc.append(weight_dict.get('Al', 0))
            Ti_calc.append(weight_dict.get('Ti', 0))

        data['Fe_calc'] = Fe_calc
        data['C_calc'] = C_calc
        data['Mn_calc'] = Mn_calc
        data['Si_calc'] = Si_calc
        data['Cr_calc'] = Cr_calc
        data['Ni_calc'] = Ni_calc
        data['Mo_calc'] = Mo_calc
        data['V_calc'] = V_calc
        data['N_calc'] = N_calc
        data['Nb_calc'] = Nb_calc
        data['Co_calc'] = Co_calc
        data['W_calc'] = W_calc
        data['Al_calc'] = Al_calc
        data['Ti_calc'] = Ti_calc

        return data