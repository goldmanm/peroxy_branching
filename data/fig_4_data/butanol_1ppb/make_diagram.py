import os
import sys
import inspect

import numpy as np
import pandas as pd

import cantera as ct

#add code folder to path so simulation can be imported
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, os.path.join(repo_dir, 'code'))

import simulation as ctt


relative_resolution = 6
temperatures = np.linspace(250,1250,int(25*relative_resolution))
pressures = np.logspace(3,7,int(15*relative_resolution))
NO = [1e-9]

def simulate_branching_various_conditions_3D(cantera_input_path, HO2_frac, conversion_species=['npropyloo','npropyl'], desired_conversion=0.95,starting_alkyl_radical = 'npropyl',
                                          expected_products = None, peroxy_frac=1e-17, 
                                          temperatures=np.linspace(250,1250,25), pressures = np.logspace(3,7,15), NO_fracs = np.logspace(-12,-4,15)):

    combs = [{'temp (K)':f1, 'NO (fraction)':f2, 'pres (Pa)':f3} for f1 in temperatures for f2 in NO_fracs for f3 in pressures]
    df = pd.DataFrame(combs)
    
    for index in df.index:
        initialMoleFractions={
            "NO": df.loc[index,'NO (fraction)'],
            "HO2": HO2_frac,
            starting_alkyl_radical: peroxy_frac,
            'O2': 0.21,
            "N2": 0.79,
        }
        solution = ctt.create_mechanism(cantera_input_path)

        conditions = df.loc[index,'temp (K)'], df.loc[index,'pres (Pa)'], initialMoleFractions
        solution.TPX = conditions
        reactor = ct.IdealGasConstPressureReactor(solution, energy='off')
        simulator = ct.ReactorNet([reactor])
        solution = reactor.kinetics
        
        peroxy_conc = sum([solution.concentrations[solution.species_index(s)] for s in conversion_species])
        outputs = ctt.run_simulation_till_conversion(solution=solution,
                                     conditions= conditions, 
                                     species = conversion_species,
                                     conversion = desired_conversion,
                                     condition_type='constant-temperature-and-pressure',
                                     output_reactions=False,
                                     atol=1e-25,
                           )
        species = outputs['species']

        for name, conc in species.iteritems():
            if  not expected_products or name in expected_products:
                df.loc[index,name] = (conc / peroxy_conc).values[-1]
            
    df.fillna(value = 0, inplace = True)
    return df

# start script
conversion_species = ['gR', 'gRO2']
starting_alkyl_radical = 'gR'
# 8 cycles
species_to_plot = ['OOCC(CO)C',
                       ['[O]CC(CO)C','galdol'], 
                       'NO2OCC(CO)C',
                       ['galdoxy','O=CC(C)C=O'],
                       'OOCC(C=O)C',
                       'galkene',
                       ['disub_c4ether','disub_epoxy','monosub_c4ether'],
                       ['CC=C','propene3ol'],
                       ['OOCC(C(O[O])O)C','O=CC(C)C(OO)O','OOCC(CO)CO[O]','OCC(O[O])(COO)C','O=CC(CO)COO','OCC(OO)(C=O)C'],
                      ]
in_legend = ['HO2 pathway', 'NO alkoxy pathway','NO nitrate pathway','water loss','KHP Cycle','HO2 + alkene', 
                 'cyclic ether + OH', 'R decomp','KHP Cycle 2',]
    
expected_products = []
for item in species_to_plot:
    if isinstance(item,list):
        for item2 in item:
            expected_products.append(item2)
    else:
        expected_products.append(item)


other_conditions = {'HO2_frac': 1e-11}
df_3d = simulate_branching_various_conditions_3D('../../mechanism/gamma_i_butanol.cti',temperatures = temperatures,
                                                               pressures = pressures,
                                                               NO_fracs = NO,
                                                               starting_alkyl_radical=starting_alkyl_radical,
                                                               expected_products = expected_products,
                                                               conversion_species=conversion_species,
                                                               **other_conditions)

df_3d.to_pickle('run_data.pic')
