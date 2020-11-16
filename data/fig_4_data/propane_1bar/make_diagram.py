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
pressures = [.5e5] 
NO = np.logspace(-12,-4,int(15*relative_resolution))

def simulate_branching_various_conditions_3D(solution, HO2_frac, conversion_species=['npropyloo','npropyl'], desired_conversion=0.95,
                                          expected_products = None, peroxy_frac=1e-17, 
                                          temperatures=np.linspace(250,1250,25), pressures = np.logspace(3,7,15), NO_fracs = np.logspace(-12,-4,15)):

    combs = [{'temp (K)':f1, 'NO (fraction)':f2, 'pres (Pa)':f3} for f1 in temperatures for f2 in NO_fracs for f3 in pressures]
    df = pd.DataFrame(combs)
    
    for index in df.index:
        initialMoleFractions={
            "NO": df.loc[index,'NO (fraction)'],
            "HO2": HO2_frac,
            'npropyl': peroxy_frac,
            'O2': 0.21,
            "N2": 0.79,
        }
        

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
                                     skip_data=1e9,
                           )
        species = outputs['species'].iloc[-1,:]

        for name, conc in species.iteritems():
            if  not expected_products or name in expected_products:
                df.loc[index,name] = conc / peroxy_conc
            
    df.fillna(value = 0, inplace = True)
    return df

# start script
conversion_species = ['npropyl','npropyloo']
starting_alkyl_radical = 'npropyl'
species_to_plot = ['npropylooh',
                       ['npropyloxy','propanal','CH3CH2OO','CH3CH2OOH'], #the c2 comes from breaking of npropyloxy
                       'npropylONO2',
                       ['prod_1','CO','prod_2','acrolein','prod_3','frag_3','propen1oxy','prod_5','vinoxy',], # 1 CO produced by the khp cycle
                       'C3H6',
                       'C2H4',
                       'propoxide']
in_legend = ['hydroperoxy', 'alkoxy','nitrate','isom','propene', 'R decomp','epoxy']
# get expected products
expected_products = []
for item in species_to_plot:
    if isinstance(item,list):
        for item2 in item:
            expected_products.append(item2)
    else:
        expected_products.append(item)

reactions_to_remove = ['CH3CH2OO + NO <=> NO2 + ethoxy',
                       'C2H4 + H (+M) <=> C2H5 (+M)',
                       '2 C2H4 <=> C2H3 + C2H5',
                       'C2H5 + CH3 <=> C2H4 + CH4',
                       'C2H5 + H <=> C2H4 + H2',
                       'C2H5 + O2 <=> C2H4 + HO2',
                       'C2H5 + O2 <=> C2H4 + HO2',
                       'C2H5 + allyl <=> C2H4 + C3H6',
                       'CH3CH2OO <=> C2H4 + HO2',
                       'CH2CH2OOH <=> CH3CH2OO',
                       'C2H5 + O2 <=> CH2CH2OOH',
                       'npropyl <=> C3H6 + H',
                       'OH + propanal <=> H2O + propionyl',
 					   'HO2 + propanal <=> H2O2 + propionyl'
                      ]
solution = ctt.create_mechanism('../../mechanism/merchant-full_w_NO.cti', remove_reaction_equations = reactions_to_remove)
other_conditions = {'HO2_frac': 1e-11}
df_3d = simulate_branching_various_conditions_3D(solution,temperatures = temperatures,
                                                               pressures = pressures,
                                                               NO_fracs = NO,
                                                               expected_products = expected_products,
                                                               **other_conditions)

df_3d.to_pickle('run_data.pic')
