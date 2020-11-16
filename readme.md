This repository contains data and code to accompany "Fate of organic peroxy radicals in the atmosphere, combustion systems, and everywhere in between" by Mark Jacob Goldman, William H Green, and Jesse H Kroll.

The goal of this repository is to help others reproduce and build off of our work.

# Structure of database

The structure of the folders in this repository are as followed:

* **code**-code to generate figures
  * *figure_3_4.ipynb* - an IPython notebook with code to generate figures 3 and 4. Requires generating data in `data\fig_4_data\`
  * *figure_5.ipynb* - an IPython notebook with code to generate figure 5.
  * *figure_6.ipynb* - an IPython notebook with code to generate figure 6. Requires RMG installation to work.
  * *carter_atkinson_NO_branching.ipynb - code to calculate the branching of NO + RO2 described in Carter and Atkinson 1989 Journal of Atmospheric Chemistry
  * *simulation.py* - methods used to simulate reaction systems in the IPython notebooks
* **data**-contains data used to make the figures
  * **mechanism** - cantera, chemkin, and RMG files for the mechanisms used in this work
    * *merchant-full_w_NO.cti* - propane model - pressure-dependent RO2
    * *gamma_i_butanol.cti* - butanol model - pressure-dependent RO2
    * *aramcomech2_modified.cti* - butane model
    * *butanol_high_p.inp* - butanol model at high pressure limit
    * *species_dictionary_butanol.txt* - file for RMG to read butanol model
  * **fig_4_data** - place for calculated data to generate figure 4. Data is generated by running the `make_diagram.py` files within this folder.
  * *license* - information on usage of code and data within this repository

# Required dependendencies

A file called environment.yml contains the list of python packages that were used in the generation of the methods, and the `conda` command can be used to generate an environment from this. In addition to this, you will need versions of [RMG-Py](https://github.com/goldmanm/RMG-Py/) and [RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database/).


