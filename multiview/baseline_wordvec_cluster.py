"""Using only unigram vectors from transcriptions of accented speech, try to cluster and do accent ID. Clustering includes: kmeans"""

import scipy
import argparse
from scipy.spatial.distance import cdist
import codecs
import time
from collections import defaultdict

__author__='Emily Ahn'

def kmeans(points, labels, k, maxiter):
    """Cluster points (named with corresponding labels) into k groups.
    Return a dictionary mapping each cluster ID (from 0 through k-1)
    to a list of the labels of the points that belong to that cluster.
    """
    #TODO: fill in
    
    # initialize first k points as clusters (centroids)
    # cluster_dict = defaultdict(list)
    # for cluster in range(k):
    #     cluster_dict[cluster] = points[cluster]

    # LOOP to account for maxiter
    this_iter = 0
    clusters = scipy.zeros(len(points))

    while (this_iter < maxiter):
        # get distances from point_i to each cluster, and assign it to the cluster with min dist
        
        if this_iter==0:
            centroids = [points[i] for i in range(k)]

        changed = False
        for point_i,point in enumerate(points) :
            dist = cdist(centroids,[point])

            min_index = scipy.argmin(dist)

            if min_index != clusters[point_i] :
                clusters[point_i] = min_index
                changed = True

        if not changed:
            break
        else :
            for centroid_i,centroid in enumerate(centroids) :
                curr_cluster = []
                for i in range(len(clusters)):
                    if centroid_i == clusters[i]:
                        curr_cluster.append(points[i])

                if not curr_cluster :
                    centroid[centroid_i] = points[centroid_i]
                else :
                    centroids[centroid_i] = scipy.mean(curr_cluster, axis=0)
        
        this_iter+=1

    cluster_dict = defaultdict(list)
    for cluster_i in range(k):
        curr_cluster = []
        
        for point_i,point in enumerate(clusters) :
            
            if cluster_i == point :
                curr_cluster.append(labels[point_i])

        cluster_dict[cluster_i] = curr_cluster

    return cluster_dict


        
def main():
    # Do not modify
    start = time.time()

    parser = argparse.ArgumentParser(description='Cluster vectors with k-means.')
    parser.add_argument('vecfile', type=str, help='name of vector file (exclude extension)')
    parser.add_argument('k', type=int, help='number of clusters')
    parser.add_argument('--maxiter', type=int, default=100, help='maximum number of k-means iterations')
    args = parser.parse_args()

    points = scipy.loadtxt(args.vecfile+'.vecs')
    labels = codecs.open(args.vecfile+'.labels', 'r', 'utf8').read().split()

    clusters = kmeans(points, labels, args.k, args.maxiter)
    outfile = args.vecfile+'.cluster'+str(args.k)
    with codecs.open(outfile, 'w', 'utf8') as o:
        for c in clusters:
            o.write('CLUSTER '+str(c)+'\n')
            o.write(' '.join(clusters[c])+'\n')

    print time.time()-start, 'seconds'

if __name__=='__main__':
    # main()
    pass