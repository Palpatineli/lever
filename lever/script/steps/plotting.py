from pypedream import draw_nx, to_nx
from classifier import build_task
from behavior import build_task_shape

def main():
    classifier_task = build_task()
    shape_task = build_task_shape()
    graph = to_nx([classifier_task, shape_task])
    draw_nx(graph)

if __name__ == '__main__':
    main()
