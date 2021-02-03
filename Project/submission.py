# Importing the required modules
import numpy as np
from scipy.spatial import distance
import copy
import heapq
from collections import defaultdict



def pq(data, P, init_centroids, max_iter):
    '''

    :param data:
    Data is an array with shape (N,M) and dtype='float32',
    where N is the number of vectors and M is the dimensionality.

    :param P: is the number of partitions/blocks the vector will be split into

    :param init_centroids:
    init_centroids is an array with shape (P,K,M/P) and dtype='float32',
    which corresponds to the initial centroids for P blocks.
    For each block, K M/P-dim vectors are used as the initial centroids.

    :param max_iter:
    The maximum number of iterations the K-medians alogrithm has to run.

    :return:
    The pq() method returns a codebook and codes for the data vectors, where

    codebooks is an array with shape (P, K, M/P) and dtype='float32',
    which corresponds to the PQ codebooks for the inverted multi-index.
    E.g., there are P codebooks and each one has K M/P-dimensional codewords.

    codes is an array with shape (N, P) and dtype=='uint8',
    which corresponds to the codes for the data vectors. The dtype='uint8' is because
    K is fixed to be 256 thus the codes should integers between 0 and 255.

    :Procedure:
    The data is split into P partitions. Then each partition along with their respective codebook
    with the max-iterations is passed into the K-medians algorithm. The alogrithm runs for 20
    iterations and for every iteration it updates the code and codebooks. After the 20th iteration the
    code are again updated using the 20th iteration codebooks to get the final codes.

    '''


    # Constants
    QUERY = "PQ"

    # Variables
    final_codes_list = []
    final_code_book = []
    sub_vector_list = np.split(data, P, axis=1)
    k_list = [i for i in range(init_centroids.shape[1])]


    for i in range(P):
        if i == 0:
            first_code_book, codes_list = k_medians(sub_vector_list[i], init_centroids[i], k_list, max_iter)
            first_codes_list = distance_cal(sub_vector_list[i], first_code_book, QUERY)

            final_codes_list.append(first_codes_list)
            final_code_book.append(first_code_book)
        else:
            temp_code_book, codes_list = k_medians(sub_vector_list[i], init_centroids[i], k_list, max_iter)
            temp_codes_list = distance_cal(sub_vector_list[i], temp_code_book, QUERY)

            final_codes_list.append(temp_codes_list)
            final_code_book.append(temp_code_book)

    final_codes = np.transpose(np.array(final_codes_list))
    final_codes = final_codes.astype('uint8')

    final_code_book = np.array(final_code_book)

    return final_code_book, final_codes


def query(queries, codebooks, codes, T=10):
    '''

    :param queries:
    The query is an array with shape (Q, M) and dtype='float32',
    where Q is the number of query vectors and M is the dimensionality.

    :param codebooks:
     is an array with shape (P, K, M/P) and dtype='float32',
     which corresponds to the codebooks returned by pq().

    :param codes:
    codes is an array with shape (N, P) and dtype=='uint8',
    which corresponds to the codes returned by pq().

    :param T:
    T is an integer which indicates the minimum number of returned candidates for each query

    :return:
    The query() method returns an array contains the candidates for each query. Candidates is a list
    with Q elements, where the i-th element is a set that contains at least T integers,
    corresponds to the id of the candidates of the i-th query.

    :Procedure:
    The query is split into required number of partitions. Then, we create a dictionary where we find
    which observation belongs to which cluster using the codes, with the help enumerate method.
    Consider only the i-th partition of the query along with the i-th codebook which is sent to the
    distance_cal method which calculates the distance from the i-th partition of query and each observation
    of the codebook and returns the costs and the codes in a sorted manner.
    Then, inside the while loop we calculate the candidates for each of the query and append it to the final
    list.
    '''

    # Constants
    QUERY = "Query"

    # Variables
    q_number = queries.shape[0]
    P = codebooks.shape[0]
    f_candidates = []

    sub_query_list = np.split(queries, P, axis=1)

    codes_dict = defaultdict(list)
    for index, data_point in enumerate(codes):
        codes_dict[tuple(data_point)].append(index)


    for k in range(q_number):
        sorted_dist = []
        cost_list = []
        cost_coor = {}
        queue = []
        ded_up = set()

        for i in range(P):
            query = np.reshape(sub_query_list[i][k], (1, -1))
            distance_list, temp_cost_list = distance_cal(query, codebooks[i], QUERY)
            sorted_dist.append(distance_list)
            cost_list.append(temp_cost_list)

        code_distance = np.transpose(np.array(sorted_dist))
        code_cost = np.transpose(np.array(cost_list))


        coor = [0 for _ in range(P)]
        first_cost = sum([code_cost[0][i] for i in range(P)])
        cost_coor[first_cost] = [coor]

        T_check = 0
        first_loop_check = True
        codes_check_list = []
        w_candidates = set()

        while T_check < T:

            if first_loop_check:
                queue.append(first_cost)
                queue, ded_up, cost_coor, coor_check = cost_neighbours(queue, ded_up, cost_coor, code_cost, P)
                first_loop_check = False

            else:
                queue, ded_up, cost_coor, coor_check = cost_neighbours(queue, ded_up, cost_coor, code_cost, P)

            uv_codes = []
            for column in range(P):
                row = coor_check[column]
                code_check = code_distance[row][column]

                uv_codes.append(code_check)

            if uv_codes not in codes_check_list:
                candidate = set(codes_dict[tuple(uv_codes)])

            w_candidates.update(candidate)

            T_check = len(w_candidates)

        f_candidates.append(w_candidates)

    return f_candidates


def k_medians(obs, code_book, k_list, iter=20):
    '''

    :param obs:
    The obs are one of the i-th partitions of the data.

    :param code_book:
    Codebooks are the centroids of the i-th partition of the data.

    :param k_list:
    K_list is a list of integers from 1 to the number of observations in the data

    :param iter:
    The maximum number of iterations the K-median Algorithm has to run

    :return:
    This returns the codes and the codebooks for the i-th partition of the data.

    :Procedure:
    First, the new codes are calculated for using the codebook obtained from the previous
    iteration or initial centroid if its the 1st iteration. The new codes are obtained from
    the distance_cal function. After getting the codes we need to Update the codebook for the
    next iteartion. The updation of the codebook is done in the update_code_book function.
    '''

    # Constants
    QUERY = "PQ"

    # Variables
    iteration = 0

    while (iteration < iter):
        # compute the codes and distances between each observation and code_book when the query == "PQ"
        obs_code = distance_cal(obs, code_book, QUERY)

        # updating and creating new code books from the given observations
        code_book = update_code_book(obs, obs_code, code_book, k_list)

        iteration += 1

    return code_book, obs_code


def distance_cal(obs, code_book, query_type):
    '''

    :param obs:
    The obs are one of the i-th partitions of the data.

    :param code_book:
    Codebooks are the centroids of the i-th partition of the data.

    :param query_type:
    Query_type is to indicate if the distance calculation is for the K-median which is
    indicate by 'PQ' or for finding the distance for the a query, indicated by "Query".

    :return:
    The return type for "PQ" is the new codes

    The return type for "Query" are codes and the distances for each for the codes in a sorted manner.

    :Procedure:
    First, the distance array is calculated using the cdist function from the scipy.spatial.distance.
    We used the mertic as "cityblock" to make our PQ algorithm run on L1 distance.

    For "PQ",
    We have used the np.argmin function to find the observation which belongs to which centroid and return the
    new codes.

    For "Query",
    We have used the np.agrsort to sort the distance array to find which centorid is the closest to our query, along
    with the cost.
    Then two separate arrays, one with codes and the other with cost of each code is returned
    '''
    distance_array = distance.cdist(obs, code_book, 'cityblock')

    if query_type == 'PQ':
        code = distance_array.argmin(axis=1)

        return code
    else:
        d_codes = np.argsort(distance_array[0])
        d_cost = np.sort(distance_array[0])

        return d_codes, d_cost


def update_code_book(obs, codes, code_book, k_list):
    '''

    :param obs:
    The obs are one of the i-th partitions of the data.

    :param code_book:
    Codebooks are the centroids of the i-th partition of the data.

    :param codes:
    Codes are the cluster number to which each of the observation belongs to.

    :param k_list:
    K_list is a list of integers from 1 to the number of observations in the data

    :return:
    This function returns the updated new codebook for the i-th partition.

    :Procedure:
    First, using the dict_list function each of the observation is assigned to the cluster
    in a dictionary. For every cluster a list is created to which each of the vector
    of the cluster is appended from the observations and then converted into an array.
    Then the median is calculated using the np.median function.

    The centroid which does not have any observation close to it will not be present in the
    cluster dictionary and hence it is calculated and stored in the missing_centroid using the
    K_list. Then the missing centroid is taken from the code_book and inserted into the new
    codebook.
    '''

    # Variables
    code_book_list = []

    cluster = dict_list(codes)
    missing_centroid = sorted(set(k_list) - set(codes))

    for j in range(code_book.shape[0]):

        if j in cluster:
            centroid_cal = []
            for i in cluster[j]:
                centroid_cal.append(obs[i])

            new_centroid = np.median(np.array(centroid_cal), axis=0)
            code_book_list.append(list(new_centroid))

    new_code_book = np.array(code_book_list)

    if len(missing_centroid) != 0:
        for i in missing_centroid:
            new_code_book = np.insert(new_code_book, i, code_book[i], 0)

    return new_code_book


def dict_list(codes):
    '''

    :param codes:
    Codes are the cluster number to which each of the observation belongs to.

    :return:
    This returns a dictionary with key as the cluster number and values as the list
    of number. Each number corresponds to the row number in the observation data.

    :Procedure:
    First, We enumerate the codes and then we used the enumerated object to assign each
    observation to the cluster which it belongs to.
    The cluster is a dictionary with key as the cluster number and value is a list with
    number corresponding to the row number in the observation data.
    '''

    enum_list = list(enumerate(codes))
    clusters = {}
    for index, cluster_no in enum_list:
        if cluster_no in clusters:
            clusters[cluster_no].append(index)
        else:
            clusters[cluster_no] = [index]

    return clusters


def cost_neighbours(queue, ded_up, cost_coor, code_cost, P):
    '''

    :param queue:
    The queue contains the costs associated with codes in a inverted multi-index

    :param ded_up:
    It is a list of list which contians the coordinates which have been visited.

    :param cost_coor:
    Cost_coor is a dictionary with cost as key and value as list of list whose coordinates
    have the cost from the inverted multi-index.

    :param code_cost:
    Code_cost consists of cost associated with each query partition and its respective codebook. The cost
    is calculated and present in each column for every partition.

    :param P:
    P is the required number of partitions

    :return:
    The function returns the queue, ded_up, cost_coor and coordinates which are updated everytime.

    :Procedure:
    The first value of the queue is taken and used as key. The value associated with the key is popped
    from the cost_coor and then assigned to coordinates to find its neighbours. If key has only one coordinate
    then the key is deleted from the cost_coor and the key is also popped from the queue. Else just the
    coordinates are obtained from the dictionary.

    Then we check if the coordinates are present in the ded_up and if not present then we find the number of
    neighbours as P since only those neighbours have to by added to queue to avoid violating the skyline
    principle. Then the cal_cost method is used to calculate the cost of each neighbour and then added to
    the cost_coor dictionary and finally if a new key is added then the key is added to the queue and then
    queue is heapified and coordinates are appended to the ded_up.
    '''

    key = queue[0]  # Getting the first cost from the queue

    if len(cost_coor[key]) == 1:
        coordinates = cost_coor[key].pop(0)  # Getting the coordinates of the first element in queue
        del cost_coor[key]  # Deleting the key from the dictionary
        heapq.heappop(queue)
    else:
        coordinates = cost_coor[key].pop(0)

    if tuple(coordinates) not in ded_up:
        for i in range(P):
            new_coordinates = copy.deepcopy(coordinates)
            new_coordinates[i] = coordinates[i] + 1

            neigh_cost = cal_cost(new_coordinates, code_cost)  # Calculating cost of new neighbour

            if neigh_cost >= 0:
                if neigh_cost in cost_coor:
                    cost_coor[neigh_cost].append(new_coordinates)
                else:
                    cost_coor[neigh_cost] = [new_coordinates]
                    queue.append(neigh_cost)


    heapq.heapify(queue)
    ded_up.add(tuple(coordinates))  # appending the coordinates to dedup to make we have already visisted

    return queue, ded_up, cost_coor, coordinates


def cal_cost(new_coordinates, code_cost):
    '''

    :param new_coordinates:
    These are coordinates of the each neighbour whose cost has to be calculated.

    :param code_cost:
    Code_cost consists of cost associated with each query partition and its respective codebook. The cost
    is calculated and present in each column for every partition.

    :return:
    This returns the calculated cost for the coordinates associated with the each code.

    :Procedure:
    A variable cost is created to calculate the cost of each of coordinate from the code_cost with
    each element of coordinate being the row of code_cost and column being the partition of code_cost.
    If an out of bound coordinate element is obtained then IndexError might occur and hence to prevent that
    we used try and catch. In such a case -1 is returned as cost.
    '''

    # Variables
    cost = 0

    try:
        for column in range(len(new_coordinates)):
            row = new_coordinates[column]
            cost += code_cost[row][column]
    except IndexError:
        cost = -1

    return cost
