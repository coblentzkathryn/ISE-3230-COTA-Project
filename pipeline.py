"""
Complete pipeline:
- downloads GTFS (COTA) stops
- downloads TIGER tracts (Ohio)
- queries ACS for population
- builds candidate stops
- computes distances (Haversine by default)
- outputs CSVs ready for optimization (Gurobi MILP)
"""


# ---------------- IMPORTS ----------------
import os, zipfile, io, tempfile
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm


# Optional
try:
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except:
    HAVE_SKLEARN = False


# ---------------- Variables & Sources ------------
GTFS_URL = "https://www.cota.com/data/cota.gtfs.zip"
TIGER_TRACTS_URL = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_39_tract.zip"
ACS_YEAR = "2023"
ACS_VARIABLES = ["B01003_001E"]  # total population
CENSUS_API = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"


# ---------------- MILP Variables ------------------
FRANKLIN_FIPS = ("39","049")
MAX_NEW_STOPS = 5
D_MAX = 805  # meters (~1/2 mile)
DIST_METHOD = "euclidean"
MAX_CANDIDATES = 200
OUT_DIR = "output"
METRIC_CRS = 3857


os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- UTILITY FUNCTIONS ----------------
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians,[lon1,lat1,lon2,lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return 6371000*c


def ensure_active_geometry(gdf, geom_name='geometry', crs_epsg=4326):
    if geom_name not in gdf.columns:
        raise ValueError(f"No column named {geom_name}")
    if gdf._geometry_column_name != geom_name:
        gdf = gdf.set_geometry(geom_name)
    gdf.set_crs(epsg=crs_epsg, inplace=True, allow_override=True)
    return gdf


# ---------------- MAIN PIPELINE ----------------
def main():
    # 1. Download GTFS stops
    print("Downloading GTFS stops...")
    r = requests.get(GTFS_URL, stream=True, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    stops_df = pd.read_csv(z.open("stops.txt"))
    if 'stop_lat' not in stops_df.columns or 'stop_lon' not in stops_df.columns:
        raise RuntimeError("GTFS stops.txt missing stop_lat/stop_lon")
    stops_gdf = gpd.GeoDataFrame(stops_df,
        geometry=gpd.points_from_xy(stops_df['stop_lon'], stops_df['stop_lat']),
        crs='EPSG:4326')
    stops_gdf = ensure_active_geometry(stops_gdf)


    # 2. Download TIGER tracts
    print("Downloading TIGER tracts...")
    r2 = requests.get(TIGER_TRACTS_URL, stream=True, timeout=60)
    r2.raise_for_status()
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir,"tracts.zip")
    with open(zip_path,"wb") as fw:
        fw.write(r2.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir)
    tracts_all = gpd.read_file(tmpdir, ignore_geometry=False).to_crs(epsg=4326)
    tracts_fran = tracts_all[(tracts_all['STATEFP']==FRANKLIN_FIPS[0]) &
                             (tracts_all['COUNTYFP']==FRANKLIN_FIPS[1])].copy()
    tracts_metric = tracts_fran.to_crs(epsg=METRIC_CRS)
    tracts_metric['centroid_geom'] = tracts_metric.geometry.centroid
    tracts_centroids = tracts_metric.set_geometry('centroid_geom').to_crs(epsg=4326).reset_index(drop=True)
    tracts_centroids = ensure_active_geometry(tracts_centroids)


    # 3. Query ACS population
    print("Querying ACS for population...")
    params = {"get": ",".join(["NAME"]+ACS_VARIABLES),
              "for":"tract:*",
              "in":f"state:{FRANKLIN_FIPS[0]} county:{FRANKLIN_FIPS[1]}"}
    resp = requests.get(CENSUS_API, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df_acs = pd.DataFrame(data[1:], columns=data[0])
    df_acs = df_acs.rename(columns={ACS_VARIABLES[0]:"population"})
    df_acs['GEOID'] = df_acs['state']+df_acs['county']+df_acs['tract']
    df_acs['population'] = pd.to_numeric(df_acs['population'], errors='coerce').fillna(0).astype(int)
    tracts_centroids = tracts_centroids.merge(df_acs[['GEOID','population']], on='GEOID', how='left')
    tracts_centroids['population'] = tracts_centroids['population'].fillna(0).astype(int)


    # 4. Prepare demand points
    print("Preparing demand points...")
    demand = tracts_centroids[tracts_centroids['population']>0].copy().reset_index(drop=True)
    demand['geometry'] = demand['geometry'].apply(lambda g: g if g.geom_type=='Point' else g.centroid)
    demand = demand[~demand['geometry'].is_empty].copy().reset_index(drop=True)
    demand['lon'] = demand.geometry.x
    demand['lat'] = demand.geometry.y
    demand = ensure_active_geometry(demand)


    # 5. Compute distance to nearest stop
    print("Computing distance to nearest existing stop...")
    stops_coords = list(zip(stops_gdf.geometry.y, stops_gdf.geometry.x))
    def nearest_stop_distance(lat,lon):
        return min(haversine(lat,lon,s[0],s[1]) for s in stops_coords)
    demand['dist_to_existing_m'] = demand.apply(lambda r: nearest_stop_distance(r['lat'],r['lon']), axis=1)


    # 6. Build candidate sites
    underserved = demand[demand['dist_to_existing_m']>D_MAX].copy()
    candidates = underserved[['GEOID','geometry','lat','lon','population']].copy()
    candidates.rename(columns={'population':'cand_pop'}, inplace=True)
    candidates['i'] = range(len(candidates))


    if len(candidates) > MAX_CANDIDATES and HAVE_SKLEARN:
        print(f"Clustering {len(candidates)} candidates -> {MAX_CANDIDATES}")
        coords = np.vstack([candidates['lon'].values, candidates['lat'].values]).T
        kmeans = KMeans(n_clusters=MAX_CANDIDATES, random_state=0).fit(coords)
        centers = kmeans.cluster_centers_
        cand_df = pd.DataFrame(centers, columns=['lon','lat'])
        cand_df['i'] = range(len(cand_df))
        cand_df['geometry'] = gpd.points_from_xy(cand_df.lon, cand_df.lat)
        candidates = gpd.GeoDataFrame(cand_df, geometry='geometry', crs='EPSG:4326')


    print(f"Number of candidate locations: {len(candidates)}")


    # 7. Build coverage matrix
    print("Building coverage matrix...")
    coverage = []
    for idx_d, row_d in tqdm(demand.iterrows(), total=len(demand)):
        covered = []
        for idx_c, row_c in candidates.iterrows():
            dist = haversine(row_d['lat'], row_d['lon'], row_c['lat'], row_c['lon'])
            if dist <= D_MAX:
                covered.append(row_c['i'])
        coverage.append((row_d.name, covered))


    return candidates, demand, coverage


# ---------------- Main ----------------
if __name__=="__main__":
    candidates, demand, coverage = main()
    # TL:DR - Will write candidates, coverage, and demand as a CSV in output folder
    # Save candidates
    candidates[['i','lat','lon']].to_csv('output/candidates.csv', index=False)
    # Save demand points
    demand[['GEOID','lat','lon','population']].rename(columns={'GEOID':'id'}).to_csv('output/demand.csv', index=False)
    # Save coverage
    cov_rows = []
    for d, covered in coverage:
        for j in covered:
            cov_rows.append({'demand_id': d, 'candidate_id': j, 'covers': 1})
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv('output/coverage.csv', index=False)


    print("Output files saved to the 'output' folder.")
    print(f"Saved {len(candidates)} candidates")
    print(f"Saved {len(demand)} demand points")
    print(f"Saved {len(cov_df)} coverage entries")
