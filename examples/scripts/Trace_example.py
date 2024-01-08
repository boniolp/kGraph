import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

sys.path.insert(1, '../../utils/')
from utils import fetch_ucr_dataset_online

from kgraph import kGraph



if __name__ == '__main__':
	
	X, y = 	fetch_ucr_dataset_online("Coffee")
	# Executing kGraph

	clf = kGraph(n_clusters=len(set(y)),n_lengths=10,n_jobs=4)
	clf.fit(X)

	print("Selected length: ",clf.optimal_length)
	print("ARI score: ",adjusted_rand_score(clf.labels_,y))

	# Plotgraphoid

	clf.show_graphoids(group=True,save_fig=True,namefile='Trace_kgraph')

	# Plot representative patterns for each cluster

	nb_patterns = 1

	nodes = clf.interprete(nb_patterns=nb_patterns)

	plt.figure(figsize=(10,4*nb_patterns))
	count = 0
	for j in range(nb_patterns):
		for i,node in enumerate(nodes.keys()):
			count += 1
			plt.subplot(nb_patterns,len(nodes.keys()),count)
			mean,sup,inf = clf.get_node_ts(X=X,node=nodes[node][j][0])
			plt.fill_between(x=list(range(int(clf.optimal_length))),y1=inf,y2=sup,alpha=0.2) 
			plt.plot(mean,color='black')
			plt.plot(inf,color='black',alpha=0.6,linestyle='--')
			plt.plot(sup,color='black',alpha=0.6,linestyle='--')
			plt.title('node {} for cluster {}: \n (representativity: {:.3f} \n exclusivity : {:.3f})'.format(nodes[node][j][0],node,nodes[node][j][3],nodes[node][j][2]))
	plt.tight_layout()

	plt.savefig('Trace_cluster_interpretation.jpg')
	plt.close()

