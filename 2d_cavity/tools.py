import math
import numpy as np
import matplotlib.pyplot as plt


def LHSample(D, bounds, N):
    # """
    # :param D: Number of parameters
    # :param bounds:  [[min_1, max_1],[min_2, max_2],[min_3, max_3]](list)
    # :param N: Number of samples
    # :return: Samples
    # """
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # Stretching the sampling
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Wrong value bound')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result),
           lower_bounds,
           out=result)
    return result

def distance(p1, p2):
    "return the distance between two points"
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def minDistance(pt, pts2):
    "return the min distance between one point and a set of points"
    dists = [distance(pt, i) for i in pts2]
    return min(dists)

def sort_pts(pts1, pts2, flag_reverse=False):
    "sort a set of points based on their distances to another set of points"
    minDists = []
    for pt in pts1:
        minDists.append( minDistance(pt, pts2) )
    minDists = np.array(minDists).reshape(1,-1)
    
    dists_sorted = np.sort(minDists).reshape(-1,1)
    sort_index = np.argsort(minDists)
    if flag_reverse:
        sort_index = sort_index.reshape(-1,1)
        sort_index = sort_index[::-1].reshape(1,-1)
        dists_sorted = dists_sorted[::-1]
    pts1_sorted = pts1[sort_index,:]
    pts1_sorted = np.squeeze(pts1_sorted)
    return pts1_sorted, dists_sorted


   
    
    
    
    
    
    
    
