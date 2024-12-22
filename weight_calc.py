test_str: str = "Fe0.760C0.000454Mn0.000992Si0.0640Cr0.000105Ni0.172Mo0.000114V0.000107Nb0.0000587Co0.0000925Al0.00101Ti0.00125"

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

test_list = list(test_str)

flip = False
formula_list = []
k = 0

if test_list[1].isalpha() == True:
    formula_list.append(test_list[0] + test_list[1])

elif test_list[1].isalpha == False:
    formula_list.append(test_list[0])

for i in range(0, len(test_list)):
    if flip == False:
        k += 1
        if test_list[i].isalpha() == False:
            flip = True

            if k == 1:
                formula_list.append(test_list[i-1])

            elif k == 2:
                formula_list.append(test_list[i-2] + test_list[i-1])
            
            k = 0
            final_element = i
    
    elif flip == True:
        k += 1
        if test_list[i].isalpha() == True:
            flip = False

            intermediate = ''
            for j in range(i - k, i):
                intermediate += test_list[j]
            formula_list.append(intermediate)

            k = 0

intermediate = ''
for i in range(final_element, len(test_list)):
    intermediate += test_list[i]
formula_list.append(intermediate)

formula_tuple_list: list[tuple[str, float]] = []

for i in range(0, len(formula_list), 2):
    formula_tuple_list.append(tuple((formula_list[i], float(formula_list[i+1]))))

formula_dict = dict(formula_tuple_list)

weight_dict_intermediate: dict[str, float] = {}

for key in formula_dict:
    weight_dict_intermediate[key] = formula_dict[key] * ELEMENT_WEIGHT[key]


total_weight = sum(weight_dict_intermediate.values())

weight_dict = {}
for key in weight_dict_intermediate:
    weight_dict[key] = round(100 * weight_dict_intermediate[key] / total_weight, 2)

print(weight_dict)

weight_tuple = [(x, v) for x, v in weight_dict.items()]

print(weight_tuple)
