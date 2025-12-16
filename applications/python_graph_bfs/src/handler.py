import igraph
import time

def graph_ops(size):
    graph = igraph.Graph.Barabasi(size, 10)
    return graph.bfs(0)

def handler(event, context=None):
    size = 1000
    time1 = time.time()
    result = graph_ops(size)
    time2 = time.time()
    cost = time2 - time1
    return {
        "result": "{} size graph BFS finished!".format(size),
        "cost": cost
    }


if __name__ == "__main__":
    event = {}
    print(handler(event))