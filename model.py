# %pip install pandas geopandas shapely numpy scikit-learn tqdm folium requests osmnx networkx
# %pip install gurobipy


import pandas as pd
import geopandas as gpd
import gurobipy as gp


# Tony, Wren, Kathryn - COTA Project
# Importing packages
from gurobipy import Model, GRB, quicksum


# Load data; assume same this is the same folder that has the other CSVs
candidates = pd.read_csv('output/candidates.csv')
demand = pd.read_csv('output/demand.csv')
coverage = pd.read_csv('output/coverage.csv')


# Sets
C = candidates['i'].tolist()  # candidate stop IDs
D = demand['id'].tolist()      # demand point IDs


# Build coverage dictionary: demand_id -> list of candidate_ids that cover it
coverage_dict = coverage.groupby('demand_id')['candidate_id'].apply(list).to_dict()


# Weight/demand for each demand point
w = dict(zip(demand['id'], demand['population']))


# Initialize model
m = Model("BusStopPlacement")


# Decision variables
x = m.addVars(C, vtype=GRB.BINARY, name="x")  # x_j = 1 if candidate j is selected
y = m.addVars(D, vtype=GRB.BINARY, name="y")  # y_i = 1 if demand point i is covered


# Objective: maximize total covered population
m.setObjective(quicksum(w[i]*y[i] for i in D), GRB.MAXIMIZE)


# 4a. Link y_i to coverage: demand i can only be covered if at least one covering candidate is selected
for i in D:
    covered_candidates = coverage_dict.get(i, [])
    if covered_candidates:
        m.addConstr(y[i] <= quicksum(x[j] for j in covered_candidates))
    else:
        m.addConstr(y[i] == 0)  # no candidate covers this demand


# 4b. Limit number of new stops
m.addConstr(quicksum(x[j] for j in C) == 5)  # MAX_NEW_STOPS = 5


m.optimize()


# Selected candidates
selected = [j for j in C if x[j].X > 0.5]
print("Selected candidate stops:", selected)


# Covered demand points
covered = [i for i in D if y[i].X > 0.5]
print("Total covered population:", sum(w[i] for i in covered))
