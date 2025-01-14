import numpy as np


def construct_graph(train_label_array, threshold_percentile, classes, symmetric_normalized, self_loop):
    train_label = np.nan_to_num(train_label_array)
    co_occurrence_matrix = np.zeros((train_label.shape[1], train_label.shape[1]))

    for instance in train_label:
        label_indices = np.where(instance == 1)[0]
        for i in label_indices:
            for j in label_indices:
                if i != j:
                    co_occurrence_matrix[i, j] += 1

    exists = np.sum(co_occurrence_matrix, axis=0)
    graph = (co_occurrence_matrix / (exists + 1e-10))

    if symmetric_normalized:
        g_threshold = np.percentile(graph, threshold_percentile)
        graph[graph < g_threshold] = 0
        graph[graph > g_threshold] = 1
        if self_loop:
            loop = np.eye(classes)
            graph = graph + loop
        symmetric_adjacency_matrix = np.maximum(graph, graph.T)
        row_sum = np.sum(symmetric_adjacency_matrix, axis=1)
        row_sum_sqrt = np.sqrt(row_sum)
        row_sum_sqrt[row_sum_sqrt == 0] = 1
        graph = symmetric_adjacency_matrix / (row_sum_sqrt[:, np.newaxis] * row_sum_sqrt[np.newaxis, :])

    return graph

