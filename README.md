# Insufficiently Justified Disparate Impact
Authors:
Neil L. Menghani
Edward McFowland III
Daniel B. Neill


## Building an Environment
Make sure you have Miniconda3 installed. If you prefer not to use Miniconda, you can still run the code in this repository using your Python 3.x environment of choice. Ensure that the dependencies listed in environment.yml are installed.

Open up a bash shell on your Windows, MacOS, or Linux machine

`cd` to this repository folder and use the command:

```bash
conda env create -f environment.yml
conda activate ijdi
```

Launch Jupyter Lab or, alternatively, your Python environment of choice.

```bash
jupyter lab
```


## Scripts
The following files, found in the `scripts` directory, are used by the experiments.

`prep.py`

`scan.py`

`subset.py`

`aggregate.py`

`q.py`



## Datasets
The following datasets are used by the experiments. See Appendix A.4 for a more detailed description.


COMPAS dataset (Original source: ProPublica; Pre-processed Data from: https://github.com/propublica/compas-analysis)

`compas.csv` (in `compas` directory)


German Credit (Original source: Institut f”ur Statistik und “Okonometrie; Pre-processed Data from: https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk?resource=download)

`german_credit.csv` (in `german_credit` directory)


## Running COMPAS Experiments
Run the following notebooks, found in the "compas" directory, to prepare the COMPAS dataset, perform experiments, and generate plots.


Data Prep:

`compas_prep.ipynb`

Experiment 1:

`compas_sim_1_neg.ipynb`

`compas_sim_1_pos.ipynb`


Experiment 2:

`compas_sim_2_neg.ipynb`

`compas_sim_2_pos.ipynb`


Mitigation Approach 1:

`compas_mit_1_neg.ipynb`

`compas_mit_1_pos.ipynb`


Mitigation Approach 2:

`compas_mit_2_neg.ipynb`

`compas_mit_2_pos.ipynb`


Generate Plots:

`compas_generate_plots.ipynb`


## Replicating the Experiments for German Credit
Run the following notebooks, found in the "german_credit" directory, to prepare the German Credit dataset, perform experiments, and generate plots.


Data Prep:

`german_credit_prep.ipynb`


Mitigation Approach 2:

`german_credit_mit_2_neg.ipynb`

`german_credit_mit_2_pos.ipynb`


Generate Plots:

`german_credit_generate_plots.ipynb`

