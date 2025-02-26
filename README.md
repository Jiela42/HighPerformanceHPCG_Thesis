# Folder Structure
## Topmost Folder
- HPCGLib: a C implementation of the HPCG pipeline along with testing and benchmarking functionalities
- Python_HPCGLib: the same as HPCGLib just in Python
- VisualLib: contains all sorts of plotting and visualizations as well as some sanity checks for correctness, this is also where the plots are stored
- Colorings: Output storing coloring information for SymGS (used by the VisualLib)
- dummy_timing_results: Used for testing runnability, should go into gitignore to be honest
- timing_results: this is where the benchmarks store the timing results and where the plotting grabs those results

## HPCGLib Src
- HPCG_versions: Different implementations for the HPCG functions, not all versions have all functions implemented
- MatrixLib: contains CSR and striped implementations of the HPCG matrix as well as the generations
- TimingLib: really just contains the timer used for benchmarking
- UtilLib: contains a bunch of useful stuff from a ceiling division to implementations of a norm

# Running the C implementation of HPCGLib
1. Create build folder in topmost folder
2. '''make'''
3. cd into testing or benchmarking
4. use one of the run files to run e.g. '''./run_AllTests'''
