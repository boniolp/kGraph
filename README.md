<p align="center">
<img width="190" src="./ressources/kGraph_logo.png"/>
</p>


<h1 align="center">kGraph</h1>
<h2 align="center">Explainable and Interpretable Graph-based time series clustering.</h2>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/boniolp/kgraph"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/boniolp/kgraph">
</p>
</div>

## kGraph in short

## References

## Getting started

First, in order to play with kGraph, please download the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/). Then modify the path in [utils.py](https://github.com/boniolp/kGraph/utils/utils.py) accordingly.

All python packages needed are listed in [requirements.txt](https://github.com/boniolp/dCAM/blob/main/requirements.txt) file and can be installed simply using the pip command: 

```(bash) 
conda env create --file environment.yml
conda activate kgraph
pip install -r requirements.txt
``` 

You can then install kgraph locally with the following command:

```(bash) 
pip install .
``` 

## Usage

We depicts below a code snippet demonstrating how to use kGraph.

```(python) 
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

sys.path.insert(1, './utils/')
from utils import fetch_ucr_dataset

from kgraph import kGraph



data = fetch_ucr_dataset('Trace')
X = np.concatenate([data['data_train'],data['data_test']],axis=0)
y = np.concatenate([data['target_train'],data['target_test']],axis=0)


# Executing kGraph
clf = kGraph(n_clusters=len(set(y)),n_lengths=10,n_jobs=4)
clf.fit(X)

print("ARI score: ",adjusted_rand_score(clf.labels_,y))
``` 
```
Running kGraph for the following length: [36, 72, 10, 45, 81, 18, 54, 90, 27, 63] 
Graphs computation done! (36.71151804924011 s) 
Consensus done! (0.03878021240234375 s) 
Ensemble clustering done! (0.0060100555419921875 s) 
ARI score:  0.986598879940902
```

### Visualization tools

We provide visualization methods to plot the graph and the identified clusters (i.e., graphoids). After running kGraph, you can run the following code to plot the graphs partioned in different clusters (grey are nodes that are not associated to a specific cluster).

```(python) 
clf.show_graphoids(group=True,save_fig=True,namefile='Trace_kgraph')
``` 
<p align="center">
<img width="190" src="./ressources/Trace_kgraph.jpg"/>
</p>

Instead of visualizing the graph, we can directly retrieve the most representative nodes for each cluster with the following code:

```(python) 
nb_patterns = 1

#Get the most representative nodes
nodes = clf.interprete(nb_patterns=nb_patterns)

plt.figure(figsize=(10,4*nb_patterns))
count = 0
for j in range(nb_patterns):
	for i,node in enumerate(nodes.keys()):

		# Get the time series for the corresponding node
		mean,sup,inf = clf.get_node_ts(X=X,node=nodes[node][j][0])
		
		count += 1
		plt.subplot(nb_patterns,len(nodes.keys()),count)
		plt.fill_between(x=list(range(int(clf.optimal_length))),y1=inf,y2=sup,alpha=0.2) 
		plt.plot(mean,color='black')
		plt.plot(inf,color='black',alpha=0.6,linestyle='--')
		plt.plot(sup,color='black',alpha=0.6,linestyle='--')
		plt.title('node {} for cluster {}: \n (representativity: {:.3f} \n exclusivity : {:.3f})'.format(nodes[node][j][0],node,nodes[node][j][3],nodes[node][j][2]))
plt.tight_layout()

plt.savefig('Trace_cluster_interpretation.jpg')
plt.close()
``` 
<p align="center">
<img width="190" src="./ressources/Trace_cluster_interpretation.jpg"/>
</p>

You may find a script containing all the code above [here](https://github.com/boniolp/kGraph/examples/scripts/Trace_example.py).



