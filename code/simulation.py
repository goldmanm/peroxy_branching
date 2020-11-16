# -*- coding: utf-8 -*-
import numpy as np
import cantera as ct
import pandas as pd
import re
import warnings
import copy

###################################
# 1a. system setup
###################################

def create_mechanism(full_model_file_path,kept_reaction_equations='all', remove_reaction_equations=None,
                     non_reactive_species = ['AR','N2','HE']):
    """
    This is a convenience method for reducing mechanisms when reading cantera
    files.

    input the full model and a list of reaction equations that you'd like to keep
    or a list of reaction equations to remove.

    This method should retain or remove all instances of the reactions

    returns a Cantera.Solution object of the mechanism with only the cooresponding
    reactions and their species.
    """
    desired_file = full_model_file_path
    spec = ct.Species.listFromFile(desired_file)
    rxns = ct.Reaction.listFromFile(desired_file)

    if remove_reaction_equations is not None:
        if isinstance(remove_reaction_equations,list):
            rxn_index = 0 
            while rxn_index < len(rxns):

                rxn_eq = rxns[rxn_index].equation

                if rxn_eq in remove_reaction_equations:
                    del rxns[rxn_index]
                else:
                    rxn_index += 1
            reduced_species = eliminate_species_from_mechanism(spec,rxns,non_reactive_species)
            return ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=reduced_species, reactions=rxns)
        else:
            raise TypeError('remove reaction equations must be a list if specified. It is currently {}'.format(remove_reaction_equations))


    if kept_reaction_equations=='all':
        return ct.Solution(full_model_file_path)
    else:
        reduced_reactions = reduce_reactions_in_mechanism(rxns,kept_reaction_equations)
        reduced_species = eliminate_species_from_mechanism(spec,reduced_reactions,non_reactive_species)
        return ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                          species=reduced_species, reactions=reduced_reactions)
    
###################################
# 1b. mechanism reduction
###################################

def reduce_reactions_in_mechanism(reaction_list, kept_reaction_equations):
    """
    finds reactions that match the form of the reaction equations in 
    kept_reaction_equations. It returns just the reactions that are meant
    to be in the mechanism.
    
    reaction_list = list of cantera Reaction objects
    kept_reaction_equations = list of strings of reaction equations to keep.
    
    This does not check for equations not in kept_reaction_equations. must be fixed
    """
    reduced_reaction_list = []
    found_reaction = np.full(len(kept_reaction_equations), False, dtype=bool)
    for reaction in reaction_list:
        if reaction.equation in kept_reaction_equations:
            reduced_reaction_list.append(reaction)
            found_reaction[kept_reaction_equations.index(reaction.equation)] = True
    if not all(found_reaction):
        reactions_missed = np.array(kept_reaction_equations)[~ found_reaction]
        raise Exception('Reactions not found in solution or appear twice in the kept_reaction_list: ' + \
                                        str(reactions_missed) + \
                                        str())
    return reduced_reaction_list

def eliminate_species_from_mechanism(species_list, kept_reactions,inert_species):
    """
    finds all the species in kept_reactions, and returns a list of species
    objects of those species. inert_species are automatically kept.
    """
    
    reacting_species = []
    for reaction in kept_reactions:
        reacting_species += list(reaction.reactants.keys()) + list(reaction.products.keys())
    # remove duplicates and add inert
    reduced_species_name_list = list(set(reacting_species)) + inert_species
    
    reduced_species_list = []
    for species in species_list:
        if species.name in reduced_species_name_list:
            reduced_species_list.append(species)
            
    return reduced_species_list

###################################
# 1c. run mechanism
###################################

def run_simulation(solution,  times, conditions=None,
                      condition_type = 'adiabatic-constant-volume',
                      output_species = True,
                      output_reactions = True,
                      output_directional_reactions = False,
                      output_rop_roc = False,
                      atol = 1e-15,
                      rtol = 1e-9,
                      temperature_values=None):
    """
    This method iterates through the cantera solution object and outputs information
    about the simulation as a pandas.DataFrame object.
    
    This method returns a dictionary with the reaction conditions data, species data,
    net reaction data, forward/reverse reaction data, and the rate of production 
    and consumption (or `None` if a variable not specified). 
    
    `solution` = Cantera.Solution object
    `conditions` = tuple of temperature, pressure, and mole fraction initial 
                species (will be deprecated. Set parameters before running)
    `times` = an iterable of times which you would like to store information in
    `condition_type` = string describing the run type
    `output_species` = output a DataFrame of species' concentrations
    `output_reactions` = output a DataFrame of net reaction rates
    `output_directional_reactions` = output a DataFrame of directional reaction rates
    `output_rop_roc` = output a DataFrame of species rates of consumption & production

    condition_types supported
    #########################
    'adiabatic-constant-volume' - assumes no heat transfer and no volume change
    'constant-temperature-and-pressure' - no solving energy equation or changing
                            rate constants
    'constant-temperature-and-volume' - no solving energy equation but allows
                            for pressure to change with reactions
    'specified-temperature-constant-volume' - the temperature profile specified
                            `temperature_values`, which corresponds to the
                            input `times`, alters the temperature right before
                            the next time step is taken. Constant volume is assumed.
    """
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
    elif condition_type == 'constant-temperature-and-pressure':
        reactor = ct.IdealGasConstPressureReactor(solution, energy='off')
    elif condition_type == 'constant-temperature-and-volume':
        reactor = ct.IdealGasReactor(solution, energy='off')
    elif condition_type == 'specified-temperature-constant-volume':
        reactor = ct.IdealGasReactor(solution, energy='off')
        if temperature_values is None:
            raise AttributeError('Must specify temperature with `temperature_values` parameter')
        elif len(times) != len(temperature_values):
            raise AttributeError('`times` (len {0}) and `temperature_values` (len {1}) must have the same length.'.format(len(times),len(temperature_values)))
    else:
        supported_types = ['adiabatic-constant-volume','constant-temperature-and-pressure',
                           'constant-temperature-and-volume','specified-temperature-constant-volume']
        raise NotImplementedError('only {0} are supported. {1} input'.format(supported_types, condition_type))
    simulator = ct.ReactorNet([reactor])
    solution = reactor.kinetics
    simulator.atol = atol
    simulator.rtol = rtol
    # setup data storage
    outputs = {}
    outputs['conditions'] = pd.DataFrame()
    if output_species:
        outputs['species'] = pd.DataFrame()
    if output_reactions:
        outputs['net_reactions'] = pd.DataFrame()
    if output_directional_reactions:
        outputs['directional_reactions'] = pd.DataFrame()
    if output_rop_roc:
        outputs['rop'] = pd.DataFrame()

    for time_index, time in enumerate(times):
        if condition_type == 'specified-temperature-constant-volume':
            solution.TD = temperature_values[time_index], solution.density
            reactor = ct.IdealGasReactor(solution, energy='off')
            simulator = ct.ReactorNet([reactor])
            solution = reactor.kinetics
            simulator.atol = atol
            simulator.rtol = rtol
            if time_index > 0:
                simulator.set_initial_time(times[time_index-1])
        simulator.advance(time)
        # save data
        outputs['conditions'] = outputs['conditions'].append(
                                get_conditions_series(simulator,reactor,solution),
                                ignore_index = True)
        if output_species:
            outputs['species'] = outputs['species'].append(
                                get_species_series(solution),
                                ignore_index = True)
        if output_reactions:
            outputs['net_reactions'] = outputs['net_reactions'].append(
                                get_reaction_series(solution),
                                ignore_index = True)
        if output_directional_reactions:
            outputs['directional_reactions'] = outputs['directional_reactions'].append(
                                get_forward_and_reverse_reactions_series(solution),
                                ignore_index = True)
        if output_rop_roc:
            outputs['rop'] = outputs['rop'].append(
                                get_rop_and_roc_series(solution),
                                ignore_index = True)

    # set indexes as time
    time_vector = outputs['conditions']['time (s)']
    for output in outputs.values():
        output.set_index(time_vector,inplace=True)

    return outputs

def run_simulation_till_conversion(solution, species, conversion,conditions=None,
                      condition_type = 'adiabatic-constant-volume',
                      output_species = True,
                      output_reactions = True,
                      output_directional_reactions = False,
                      output_rop_roc = False,
                      skip_data = 150,
                      atol = 1e-15,
                      rtol = 1e-9,):
    """
    This method iterates through the cantera solution object and outputs information
    about the simulation as a pandas.DataFrame object.

    This method returns a dictionary with the reaction conditions data, species data,
    net reaction data, forward/reverse reaction data, and the rate of production 
    and consumption (or `None` if a variable not specified) at the specified conversion value.

    `solution` = Cantera.Solution object
    `conditions` = tuple of temperature, pressure, and mole fraction initial 
                species
    `species` = a string of the species label (or list of strings) to be used in conversion calculations
    `conversion` = a float of the fraction conversion to stop the simulation at
    `condition_type` = string describing the run type, currently supports 
                'adiabatic-constant-volume' and 'constant-temperature-and-pressure'
    `output_species` = output a Series of species' concentrations
    `output_reactions` = output a Series of net reaction rates
    `output_directional_reactions` = output a Series of directional reaction rates
    `output_rop_roc` = output a DataFrame of species rates of consumption & production
    `skip_data` = an integer which reduces storing each point of data.
                    storage space scales as 1/`skip_data`
    """
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
    if condition_type == 'constant-temperature-and-pressure':
        reactor = ct.IdealGasConstPressureReactor(solution, energy='off')
    else:
        raise NotImplementedError('only adiabatic constant volume is supported')
    simulator = ct.ReactorNet([reactor])
    solution = reactor.kinetics
    simulator.atol = atol
    simulator.rtol = rtol
    # setup data storage
    outputs = {}
    outputs['conditions'] = pd.DataFrame()
    if output_species:
        outputs['species'] = pd.DataFrame()
    if output_reactions:
        outputs['net_reactions'] = pd.DataFrame()
    if output_directional_reactions:
        outputs['directional_reactions'] = pd.DataFrame()
    if output_rop_roc:
        outputs['rop'] = pd.DataFrame()

    if isinstance(species,str):
        target_species_indexes = [solution.species_index(species)]
    else: # must be a list or tuple
        target_species_indexes = [solution.species_index(s) for s in species]
    starting_concentration = sum([solution.concentrations[target_species_index] for target_species_index in target_species_indexes])

    proper_conversion = False
    new_conversion = 0
    skip_count = 1e8
    while not proper_conversion:
        error_count = 0
        while error_count >= 0:
            try:
                simulator.step()
                error_count = -1
            except:
                error_count += 1
                if error_count > 10:
                    print('Might not be possible to achieve conversion at T={0}, P={1}, with concentrations of {2} obtaining a conversion of {3} at time {4} s.'.format(solution.T,solution.P,zip(solution.species_names,solution.X), new_conversion,simulator.time))
                    raise
        new_conversion = 1-sum([solution.concentrations[target_species_index] for target_species_index in target_species_indexes])/starting_concentration
        if new_conversion > conversion:
            proper_conversion = True
        
        # save data
        if skip_count > skip_data or proper_conversion:
            skip_count = 0
            outputs['conditions'] = outputs['conditions'].append(
                                    get_conditions_series(simulator,reactor,solution),
                                    ignore_index = True)
            if output_species:
                outputs['species'] = outputs['species'].append(
                                    get_species_series(solution),
                                    ignore_index = True)
            if output_reactions:
                outputs['net_reactions'] = outputs['net_reactions'].append(
                                    get_reaction_series(solution),
                                    ignore_index = True)
            if output_directional_reactions:
                outputs['directional_reactions'] = outputs['directional_reactions'].append(
                                    get_forward_and_reverse_reactions_series(solution),
                                    ignore_index = True)
            if output_rop_roc:
                outputs['rop'] = outputs['rop'].append(
                                    get_rop_and_roc_series(solution),
                                    ignore_index = True)
        skip_count += 1

    # set indexes as time
    time_vector = outputs['conditions']['time (s)']
    for output in outputs.values():
        output.set_index(time_vector,inplace=True)
    return outputs

###################################
# 1d. saving data helper methods
###################################

def get_conditions_series(simulator, reactor, solution,
                          basics= ['time','temperature','pressure','density','volume','enthalpy','internal energy']):
    """
    returns the current conditions of a Solution object contianing ReactorNet
    object (simulator) as a pd.Series.
    
    simulator = the ReactorNet object of the simulation
    solution = solution object to pull values from
    basics =a list of state variables to save 
    
    The following are enabled for the conditions:
    * time
    * temperature
    * pressure
    * density
    * volume
    * cp (constant pressure heat capacity)
    * cv (constant volume heat capacity)
    * enthalpy
    """
    conditions = pd.Series()
    # add regular conditions
    if 'time' in basics:
        conditions['time (s)'] = simulator.time
    if 'temperature' in basics:
        conditions['temperature (K)'] = solution.T
    if 'pressure' in basics:
        conditions['pressure (Pa)'] = solution.P
    if 'density' in basics:
        conditions['density (kmol/m3)'] = solution.density_mole
    if 'volume' in basics:
        conditions['volume (m3)'] = reactor.volume
    if 'cp' in basics:
        conditions['heat capacity, cp (J/kmol/K)'] = solution.cp_mole
    if 'cv' in basics:
        conditions['heat capacity, cv (J/kmol/K)'] = solution.cv_mole
    if 'enthalpy' in basics:
        conditions['enthalpy (J/kg)'] = solution.enthalpy_mass
    if 'internal energy' in basics:
        conditions['internal energy (J/kg)'] = solution.int_energy_mass
    return conditions

def get_species_series(solution, species_names = 'all'):
    """
    returns a pandas.Series of the desired species' concentrations
    
    solution = the cantera.Solution object for the simulation
    species_names = list of species names to be saved (default is all)
    """
    series = pd.Series()
    if species_names=='all':
        species_recorded = solution.species_names
    else:
        species_recorded = species_names
    mole_fractions = solution.mole_fraction_dict()
    for name in species_recorded:
        try:
            series[name] = mole_fractions[name] * solution.density_mole
        except KeyError:
            series[name] = 0
            # sends warning if user typed species incorrectly
            if name not in solution.species_names:
                warnings.warn('{} is not listed in the mole fraction dictionary and may be mispelled.'.format(name))
    return series

def get_reaction_series(solution, reaction_names = 'all'):
    """
    returns a pandas.Series of the desired reactions' net rates
    
    solution = the cantera.Solution object for the simulation
    species_names = list of reaction names to be saved (default is all)
    """
    series = pd.Series()
    if reaction_names=='all':
        reaction_names = solution.reaction_equations()

    reaction_rates = __get_rxn_rate_dict(solution.reaction_equations(),solution.net_rates_of_progress)
    for name in reaction_names:
        try:
            series[name] = reaction_rates[name]
        except KeyError:
            series[name] = 0
            warnings.warn('{} is not listed in the reaction names.'.format(name))
    return series

def get_forward_and_reverse_reactions_series(solution):
    """
    This method returns a series of the forward and reverse reactions
    """
    reaction_equations = solution.reaction_equations()
    forward_reactions = pd.Series(__get_rxn_rate_dict(reaction_equations,solution.forward_rates_of_progress))
    reverse_reactions = pd.Series(__get_rxn_rate_dict(reaction_equations,solution.reverse_rates_of_progress))
    
    forward_reactions.index = pd.MultiIndex.from_product([['forward'],forward_reactions.index], names = ['direction','reaction'])
    reverse_reactions.index = pd.MultiIndex.from_product([['reverse'],reverse_reactions.index], names = ['direction','reaction'])
    
    return pd.concat([forward_reactions,reverse_reactions])

def get_rop_and_roc_series(solution):
    """
    returns rate of production and rate of consumption to dataframe (kmol/m3s)
    This data is primarily useful for quasi-steady state analysis
    """
    species = solution.species_names
    production = pd.Series(__get_rxn_rate_dict(species,solution.creation_rates))
    consumption = pd.Series(__get_rxn_rate_dict(species,solution.destruction_rates))
    
    assert isinstance(production.index,pd.Index)
    assert isinstance(consumption.index,pd.Index)
    
    production.index = pd.MultiIndex.from_product([['production'],production.index])
    consumption.index = pd.MultiIndex.from_product([['consumption'],consumption.index])
     
    return pd.concat([production,consumption])

def __get_rxn_rate_dict(reaction_equations, net_rates):
    """
    makes a dictionary out of the two inputs. If identical reactions are encountered,
    called duplicates in Cantera, the method will merge them and sum the rate together
    """
    rxn_dict = {}
    for equation, rate in zip(reaction_equations, net_rates):
        try:
            rxn_dict[equation] += rate
        except KeyError:
            rxn_dict[equation] = rate
    return rxn_dict
