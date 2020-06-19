from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
# from leuvenmapmatching import visualization as mmviz
import osmnx as ox
import geopandas as gpd
import pandas as pd
import osmread


def map_matching(map_con, df, s=0, n=3):
    trj_ids = df.trj_id.unique()
    for i in range(s,n): 
        df_path = df[df['trj_id'] == trj_ids[i]]
        df_trj = df_path[['rawlat' ,'rawlng']]
        track = []
        max_dist = 2
        for j in range(len(df_trj)):
            track.append((df_trj.iloc[j, 0], df_trj.iloc[j, 1]))
            
        while True:
            try:
                matcher = DistanceMatcher(map_con,
                                        max_dist=max_dist, max_dist_init=1000000,  # meter
                                        min_prob_norm=0.001,
                                        non_emitting_length_factor=0.75,
                                        obs_noise=50, obs_noise_ne=75,  # meter
                                        dist_noise=50,  # meter
                                        non_emitting_states=True)
                states, lastidx = matcher.match(track)
                nodes = matcher.path_pred_onlynodes
                a = states[-1]
                break
            except:
                max_dist += max_dist*0.5
                print(max_dist)
        # print('current trajectory ID: {}'.format(trj_ids[i]))
        with open("file.txt", "a+") as output:
            for j in range(len(nodes)):
                trj_data = [trj_ids[i],map_con.graph[nodes[j]][0][0], map_con.graph[nodes[j]][0][1]]
                output.write('{}\n'.format(str(trj_data)))
        print(i)

# def plot_route(map_con, matcher, filename="my_plot.png", use_osm=True):
#     mmviz.plot_map(map_con, matcher=matcher[1],
#                 use_osm=use_osm, zoom_path=True,
#                 show_labels=False, show_matching=True, show_graph=False,
#                 filename=filename)

if __name__=="__main__":
    map_con = InMemMap("osmmap", use_latlon=True, use_rtree=True, index_edges=True)
    graph = ox.load_graphml("mynetwork.graphml")
    graph_proj = ox.project_graph(graph)

    # Create GeoDataFrames
    # Approach 2
    print("start1")
    nodes, edges = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)

    nodes_proj = nodes.to_crs("EPSG:3414")
    edges_proj = edges.to_crs("EPSG:3414")

    for nid, row in nodes_proj.iterrows():
        map_con.add_node(nid, (row['lat'], row['lon']))

    # adding edges using networkx graph
    for nid1, nid2, _ in graph.edges:
        map_con.add_edge(nid1, nid2)

    """## Read the SG dataframe"""
    sg_df = pd.read_csv('sg_car.csv')
    # print(sg_df.head())
    sg_df = sg_df.sort_values(by=['trj_id', 'pingtimestamp'])
    sg_df = sg_df.drop(sg_df[(sg_df.accuracy > 3000)].index)
    trj_ids = sg_df.trj_id.unique()
    n = len(trj_ids)
    print("Start")
    map_matching(map_con, sg_df, s=14000, n=20000)

