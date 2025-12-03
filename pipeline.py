"""
Complete pipeline:
- downloads GTFS (COTA) stops
- downloads TIGER tracts (Ohio)
- queries ACS for population
- builds candidate stops
- computes distances (Haversine by default)
- outputs CSVs ready for optimization (Gurobi MILP)
- folium map
- post optimality analysis
"""


# ---------------- IMPORTS ----------------
import os
import zipfile
import io
import tempfile
import webbrowser
import requests
import pandas as pd
import geopandas as gpd
import numpy as np


# REMOVED: import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB


from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm




# ---------------- Variables & Sources ------------
GTFS_URL = "https://www.cota.com/data/cota.gtfs.zip"
TIGER_TRACTS_URL = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_39_tract.zip"
CENSUS_API = "https://api.census.gov/data/2023/acs/acs5"
ACS_VARIABLE = "B01003_001E"  # population
FRANKLIN_FIPS = ("39","049")


# ---------------- MILP Variables ------------------
MAX_NEW_STOPS = 5     # p (baseline)
D_MAX = 805           # meters (baseline)
OUT_DIR = "output"
CRS_METRIC = 3857     # metric CRS for distance queries
os.makedirs(OUT_DIR, exist_ok=True)




# ---------------- UTILITY FUNCTIONS ----------------
def download_gtfs_stops(gtfs_url):
    r = requests.get(gtfs_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    stops_df = pd.read_csv(z.open("stops.txt"))
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs="EPSG:4326",
    )
    return stops_gdf




def download_tiger_tracts(zip_url, statefp, countyfp):
    r = requests.get(zip_url)
    tmp = tempfile.mkdtemp()
    zip_path = os.path.join(tmp, "tracts.zip")
    open(zip_path,"wb").write(r.content)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    gdf = gpd.read_file(tmp)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[(gdf.STATEFP == statefp) & (gdf.COUNTYFP == countyfp)].copy()
    return gdf




def query_acs_population(state, county, variable):
    params = {
        "get": f"NAME,{variable}",
        "for": "tract:*",
        "in": f"state:{state} county:{county}"
    }
    resp = requests.get(CENSUS_API, params=params)
    resp.raise_for_status()
    data = resp.json()
    acs = pd.DataFrame(data[1:], columns=data[0])
    acs["population"] = acs[variable].astype(int)
    acs["GEOID"] = acs.state + acs.county + acs.tract
    return acs[["GEOID","population"]]




def build_coverage_matrix_from_metric(demand_m, candidates_m, radius_m):
    dem_xy = np.c_[demand_m.geometry.x, demand_m.geometry.y]
    cand_xy = np.c_[candidates_m.geometry.x, candidates_m.geometry.y]
    tree = cKDTree(cand_xy)
    neighbors = tree.query_ball_point(dem_xy, r=radius_m)
    n = len(demand_m)
    m = len(candidates_m)
    A = np.zeros((n, m), dtype=int)
    for i, neigh in enumerate(neighbors):
        if len(neigh) > 0:
            A[i, neigh] = 1
    return A








def solve_original_setcover(A, w, p, solver=None, verbose=False):
    """
    A: (n x m) binary numpy
    w: length n (population)
    p: max number of candidate stops to open
    """
    n, m = A.shape


    # Build Gurobi model
    model = gp.Model("setcover")
    model.Params.OutputFlag = 1 if verbose else 0


    # Decision variables
    x = model.addMVar(m, vtype=GRB.BINARY, name="x")
    y = model.addMVar(n, vtype=GRB.BINARY, name="y")


    # Coverage constraints: y_i <= \sum_j A[i,j] x_j
    model.addConstrs((y[i] <= A[i, :] @ x for i in range(n)), name="coverage")


    # Limit new stops: \sum_{j} x_j <= p
    model.addConstr(x.sum() <= p, name="limit_p")


    # Objective: maximize w*y
    model.setObjective(w @ y, GRB.MAXIMIZE)


    # Solve
    model.optimize()


    if model.Status != GRB.OPTIMAL:
        return None


    x_int = (x.X > 0.5).astype(int)
    y_int = (y.X > 0.5).astype(int)
    return x_int, y_int




# -----------------------------------------------------------




def create_map_and_save(stops, tracts, candidates, demand, chosen,
                        radius_m, max_new, p_weight, scenario_idx, out_dir):
    try:
        import folium
    except Exception:
        print("folium not available, skipping map creation.")
        return


    center = [demand.geometry.y.mean(), demand.geometry.x.mean()]
    m = folium.Map(center, zoom_start=11, tiles="cartodbpositron")


    if "pop_density" in tracts.columns:
        folium.Choropleth(
            geo_data=tracts.to_json(),
            data=tracts,
            columns=["GEOID","pop_density"],
            key_on="feature.properties.GEOID",
            fill_color="YlOrRd",
            fill_opacity=0.6,
            line_opacity=0.2,
            legend_name="Population Density"
        ).add_to(m)


    for _, r in stops.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=2, color="gray", fill=True).add_to(m)


    for _, r in candidates.iterrows():
        lat, lon = r.geometry.y, r.geometry.x
        is_chosen = int(r.chosen) == 1 if "chosen" in candidates.columns else False
        color = "red" if is_chosen else "blue"
        radius = 8 if is_chosen else 4
        popup = folium.Popup(
            f"GEOID:{r.get('GEOID', '')}<br>pop:{r.get('population', '')}<br>chosen:{int(is_chosen)}",
            max_width=250
        )
        folium.CircleMarker([lat, lon], radius=radius, color=color, fill=True, popup=popup).add_to(m)
        if is_chosen:
            folium.Circle([lat, lon], radius=radius_m, color=color, fill=False, opacity=0.4).add_to(m)


    safe_d  = str(radius_m).replace('.', '')
    safe_pw = str(p_weight).replace('.', '')
    fname = f"map_s{scenario_idx}_d{safe_d}_n{max_new}_pw{safe_pw}.html"
    out_file = os.path.join(out_dir, fname)
    m.save(out_file)
    print(f"Map saved → {out_file}")
    if scenario_idx == 0:
        webbrowser.open(f"file://{os.path.abspath(out_file)}")


# Printing post optimal maps for potential future use
def create_postoptimal_maps(stops, tracts, candidates, demand, out_dir):
    """
    Reads the post_optimality_equity.csv file and generates
    HTML maps for each scenario stored inside the CSV.
    Mimics the style of create_map_and_save().
    """


    # Load scenario results
    results_path = os.path.join(out_dir, "post_optimality_equity.csv")
    if not os.path.exists(results_path):
        print("No post-optimal results found at:", results_path)
        return


    post_df = pd.read_csv(results_path)
    if post_df.empty:
        print("Post-optimal CSV is empty.")
        return


    print("\nGenerating HTML maps for all post-optimality scenarios...")


    # Need folium
    try:
        import folium
    except Exception:
        print("folium is not installed; cannot create post-optimality maps.")
        return


    # Create a fresh metric projection for candidates for rebuild of A
    candidates_m = candidates.to_crs(CRS_METRIC)
    demand_m = demand.to_crs(CRS_METRIC)


    scenario_number = 1


    for _, row in post_df.iterrows():
        dmax = row["d_max"]
        pmax = row["max_new"]
        eq = row["equity_mult"]


        # Rebuild A for the scenario
        A_s = build_coverage_matrix_from_metric(demand_m, candidates_m, dmax)


        # Rebuild scenario weights
        pop = demand["population"].values.astype(float)
        if "pop_density" in demand.columns:
            density = demand["pop_density"].fillna(0).values.astype(float)
        else:
            density = np.zeros(len(demand), dtype=float)


        if density.max() > density.min():
            density_norm = (density - density.min()) / (density.max() - density.min())
        else:
            density_norm = np.zeros_like(density)


        weights = pop * (1.0 + eq * density_norm)


        # Solve MILP again (same logic as main pipeline)
        sol_s = solve_original_setcover(A_s, weights, pmax, verbose=False)
        if sol_s is None:
            print(f" Scenario {scenario_number}: FAILED to solve")
            scenario_number += 1
            continue


        x_sel, y_sel = sol_s


        # Update candidate flags
        cand_scen = candidates.copy()
        cand_scen["chosen"] = x_sel
        chosen = cand_scen[cand_scen["chosen"] == 1].copy()


        # Create map for this scenario (mimics create_map_and_save)
        center = [demand.geometry.y.mean(), demand.geometry.x.mean()]
        m = folium.Map(center, zoom_start=11, tiles="cartodbpositron")


        # Choropleth
        if "pop_density" in tracts.columns:
            folium.Choropleth(
                geo_data=tracts.to_json(),
                data=tracts,
                columns=["GEOID","pop_density"],
                key_on="feature.properties.GEOID",
                fill_color="YlOrRd",
                fill_opacity=0.6,
                line_opacity=0.2,
                legend_name="Population Density"
            ).add_to(m)


        # Existing stops
        for _, r in stops.iterrows():
            folium.CircleMarker([r.geometry.y, r.geometry.x],
                                radius=2, color="gray", fill=True).add_to(m)


        # Candidate stops
        for _, r in cand_scen.iterrows():
            lat, lon = r.geometry.y, r.geometry.x
            is_chosen = r.chosen == 1
            color = "red" if is_chosen else "blue"
            radius = 8 if is_chosen else 4
            popup = folium.Popup(
                f"GEOID:{r.get('GEOID','')}<br>"
                f"pop:{r.get('population','')}<br>"
                f"chosen:{int(is_chosen)}",
                max_width=250,
            )
            folium.CircleMarker([lat, lon], radius=radius, color=color, fill=True,
                                popup=popup).add_to(m)


            if is_chosen:
                folium.Circle([lat, lon], radius=dmax,
                              color=color, fill=False, opacity=0.4).add_to(m)


        # Output filename
        safe_d = str(dmax).replace(".", "")
        safe_eq = str(eq).replace(".", "")
        fname = f"post_map_s{scenario_number}_d{safe_d}_n{pmax}_eq{safe_eq}.html"
        out_path = os.path.join(out_dir, fname)


        m.save(out_path)
        print(f" Scenario {scenario_number}: map saved → {out_path}")


        scenario_number += 1


    print("\nPost-optimality maps generated.")






# ---------------- MAIN PIPELINE ----------------
def main():
    print("Downloading GTFS stops...")
    stops = download_gtfs_stops(GTFS_URL)


    print("Downloading TIGER tracts and querying ACS population...")
    tracts = download_tiger_tracts(TIGER_TRACTS_URL, FRANKLIN_FIPS[0], FRANKLIN_FIPS[1])
    acs = query_acs_population(FRANKLIN_FIPS[0], FRANKLIN_FIPS[1], ACS_VARIABLE)
    tracts = tracts.merge(acs, on="GEOID", how="left")
    tracts["population"] = tracts["population"].fillna(0).astype(int)


    tracts_m = tracts.to_crs(CRS_METRIC)
    tracts["pop_density"] = tracts["population"] / (tracts_m.area / 1e6)
    centroids_m = tracts_m.geometry.centroid
    centroids_wgs84 = centroids_m.to_crs("EPSG:4326")


    demand = tracts.copy()
    demand["geometry"] = centroids_wgs84.values
    demand["lat"] = demand.geometry.y
    demand["lon"] = demand.geometry.x
    demand = gpd.GeoDataFrame(demand, geometry="geometry", crs="EPSG:4326")
    n_dem = len(demand)
    print(f"   Demand points (tract centroids): {n_dem}")


    stops_m = stops.to_crs(CRS_METRIC)
    demand_m = demand.to_crs(CRS_METRIC)


    print("Distance to nearest existing stop...")
    if len(stops_m) == 0:
        demand["dist_to_existing_m"] = np.inf
    else:
        stop_coords = np.c_[stops_m.geometry.x.values, stops_m.geometry.y.values]
        tree = cKDTree(stop_coords)
        demand_coords = np.c_[demand_m.geometry.x.values, demand_m.geometry.y.values]
        dists, idxs = tree.query(demand_coords, k=1)
        demand["dist_to_existing_m"] = dists


    underserved = demand[demand.dist_to_existing_m > D_MAX].copy()
    print(f"   Underserved demand points (baseline D_MAX={D_MAX} m): {len(underserved)}")


    candidates = underserved.reset_index(drop=True).copy()
    candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs="EPSG:4326")
    m_cand = len(candidates)
    print(f"   Candidate stop locations: {m_cand}")
    if m_cand == 0:
        print("No candidate locations (no underserved tracts). Exiting.")
        return


    candidates_m = candidates.to_crs(CRS_METRIC)
    demand_m = demand.to_crs(CRS_METRIC)


    print("Building baseline coverage matrix (KDTree)...")
    A_baseline = build_coverage_matrix_from_metric(demand_m, candidates_m, D_MAX)
    w = demand["population"].values.astype(int)


    print("Solving MILP with Gurobi...")
    sol = solve_original_setcover(A_baseline, w, MAX_NEW_STOPS, verbose=True)
    if sol is None:
        raise RuntimeError("Solver failed to find a solution for the base MILP.")
    x_sel, y_sel = sol


    candidates["chosen"] = x_sel
    demand["covered"] = y_sel


    chosen = candidates[candidates["chosen"] == 1].copy()
    total_covered_pop = int((w * y_sel).sum())
    print("\nSELECTED NEW BUS STOPS (base):")
    if not chosen.empty:
        print(chosen[["GEOID","lat","lon","population"]])
    else:
        print("  (no candidates selected)")


    print(f"\nTotal population newly covered (approx): {total_covered_pop:,}")


    candidates[["GEOID","lat","lon","chosen"]].to_csv(os.path.join(OUT_DIR,"candidates.csv"), index=False)
    demand[["GEOID","lat","lon","population","dist_to_existing_m","covered"]].to_csv(os.path.join(OUT_DIR,"demand_coverage.csv"), index=False)


    try:
        import folium
        create_map_and_save(stops, tracts, candidates, demand, chosen,
                            radius_m=D_MAX, max_new=MAX_NEW_STOPS,
                            p_weight=1.0, scenario_idx=0, out_dir=OUT_DIR)
    except Exception as e:
        print("Skipping base map creation:", e)


    print("\n\n=== POST-OPTIMALITY ANALYSIS: exploring parameters ===")


    D_MAX_LIST = [600, 805, 1000]
    MAX_NEW_LIST = [3, 5, 7]
    EQUITY_MULTIPLIERS = [0.0, 1.0, 2.0]


    if "pop_density" in demand.columns:
        density = demand["pop_density"].fillna(0).values.astype(float)
    else:
        density = np.zeros(len(demand), dtype=float)


    if density.max() > density.min():
        density_norm = (density - density.min()) / (density.max() - density.min())
    else:
        density_norm = np.zeros_like(density)


    results = []


    def run_scenario_rebuild_A_and_solve_with_weights(dmax, max_new, weights):
        A_s = build_coverage_matrix_from_metric(demand_m, candidates_m, dmax)


        sol_s = solve_original_setcover(A_s, weights, max_new, verbose=False)
        if sol_s is None:
            return None
        x_s, y_s = sol_s
        covered_pop = float((weights * y_s).sum())
        return int(round(covered_pop)), x_s, y_s


    scen_idx = 1
    for d in D_MAX_LIST:
        for p in MAX_NEW_LIST:
            for e in EQUITY_MULTIPLIERS:
                weights = demand["population"].values * (1.0 + e * density_norm)


                res = run_scenario_rebuild_A_and_solve_with_weights(d, p, weights)
                if res is None:
                    print(f"Scenario d={d} p={p} e={e} failed to solve.")
                    continue
                cov, xvec, yvec = res
                results.append(dict(d_max=d, max_new=p, equity_mult=e,
                                    coverage=cov, num_stops=int(xvec.sum())))
                print(f" Scenario {scen_idx}: d_max={d}, p={p}, equity={e} => "
                      f"covered_pop={cov}, stops_used={int(xvec.sum())}")


                cand2 = candidates.copy()
                cand2["chosen"] = xvec
                chosen2 = cand2[cand2.chosen == 1].copy()
                try:
                    create_map_and_save(stops, tracts, cand2, demand, chosen2,
                                        radius_m=d, max_new=p, p_weight=e,
                                        scenario_idx=scen_idx, out_dir=OUT_DIR)
                except Exception as e_map:
                    print(f"  Skipping map for scenario {scen_idx}:", e_map)


                scen_idx += 1


    post_df = pd.DataFrame(results)
    post_df.to_csv(os.path.join(OUT_DIR, "post_optimality_equity.csv"), index=False)
    print("\nPost-optimality results saved → output/post_optimality_equity.csv")




    # ---------------- BEST POST-OPTIMALITY SCENARIO & TOP 5 ----------------
    
    # Sort by coverage and take top 5
    top_post = post_df.sort_values("coverage", ascending=False).head(5).reset_index(drop=True)


    print("\n=== TOP 5 POST-OPTIMALITY SCENARIOS (simplified) ===")
    for i, r in top_post.iterrows():
        print(f"\n{i+1}. d_max={r.d_max}, max_new={r.max_new}, equity_mult={r.equity_mult}")
        print(f"   coverage={int(r.coverage)}, stops_used={int(r.num_stops)}")
    
    # ---- Best post-optimal scenario ----
    best = top_post.iloc[0]
    dmax_best = best.d_max
    p_best = best.max_new
    eq_best = best.equity_mult
    
    weights_best = demand["population"].values * (1.0 + eq_best * density_norm)
    A_best = build_coverage_matrix_from_metric(demand_m, candidates_m, dmax_best)
    sol_best = solve_original_setcover(A_best, weights_best, p_best, verbose=False)
    
    if sol_best is not None:
        x_best, y_best = sol_best
        cand_best = candidates.copy()
        cand_best["chosen"] = x_best
        chosen_best = cand_best[cand_best.chosen == 1].copy()
    
        print("\nSELECTED NEW BUS STOPS (post-optimal best scenario):")
        print(chosen_best[["GEOID","lat","lon","population"]])
    
        # Create and automatically open HTML map for best scenario
        try:
            create_map_and_save(
                stops, tracts, cand_best, demand, chosen_best,
                radius_m=dmax_best, max_new=p_best, p_weight=eq_best,
                scenario_idx=0, out_dir=OUT_DIR
            )
        except Exception as e_map:
            print("Skipping best scenario map creation:", e_map)




        
    print("\nCreating post-optimality HTML maps...")
    create_postoptimal_maps(stops, tracts, candidates, demand, out_dir=OUT_DIR)


# ---------------------------------------------------------
if __name__ == "__main__":
    main()

