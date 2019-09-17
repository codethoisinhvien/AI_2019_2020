
import util


def traverse_path(parent_node,parent_child_map,opt):
    print  parent_node
    print (parent_child_map)
    direction_list = []
    while True:
        map_row = parent_child_map[parent_node]
        if (len(map_row) == opt):
            parent_node = map_row[0]
            direction = map_row[1]
            direction_list.append(direction)
        else:
            break
    print ("list",direction_list)
    return direction_list
class SearchProblem:

    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):

        util.raiseNotDefined()

    def getSuccessors(self, state):

        util.raiseNotDefined()

    def getCostOfActions(self, actions):

        util.raiseNotDefined()


def tinyMazeSearch(problem):

    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):

    loc_stack = util.Queue()  # B(n)
    visited_node = {}
    parent_child_map = {}

    start_node = problem.getStartState()  # start state
    parent_child_map[start_node] = []
    loc_stack.push(start_node)


    while (loc_stack.isEmpty() == False):

        parent_node = loc_stack.pop()
        # achieve goals
        if (problem.isGoalState(parent_node)):
            pathlist = traverse_path(parent_node,parent_child_map,2)
            pathlist.reverse()
            return pathlist
        # not achieve goals
        elif (visited_node.has_key(parent_node) == False):
            visited_node[parent_node] = []
            print (visited_node)
            sucessor_list = problem.getSuccessors(parent_node)  # create children node
            no_of_child = len(sucessor_list)
            if (no_of_child > 0):
                temp = 0
                while (temp < no_of_child):
                    child_nodes = sucessor_list[temp]
                    child_state = child_nodes[0]; # position
                    child_action = child_nodes[1]; # plan action

                    #if (visited_node.has_key(child_state) == False):
                    loc_stack.push(child_state)
                    if (visited_node.has_key(child_state) == False):
                       parent_child_map[child_state] = [parent_node, child_action] # add child state

                    temp = temp + 1



def breadthFirstSearch(problem):
    loc_stack = util.Stack()  # B(n)
    visited_node = {}
    parent_child_map = {}

    start_node = problem.getStartState()  # start state
    parent_child_map[start_node] = []
    loc_stack.push(start_node)

    while (loc_stack.isEmpty() == False):

        parent_node = loc_stack.pop()
        # achieve goals
        if (problem.isGoalState(parent_node)):
            pathlist = traverse_path(parent_node, parent_child_map,2)
            pathlist.reverse()
            return pathlist
        # not achieve goals
        elif (visited_node.has_key(parent_node) == False):
            visited_node[parent_node] = []
            sucessor_list = problem.getSuccessors(parent_node)  # create children node
            no_of_child = len(sucessor_list)
            if (no_of_child > 0):
                temp = 0
                while (temp < no_of_child):
                    child_nodes = sucessor_list[temp]
                    child_state = child_nodes[0];  # position
                    child_action = child_nodes[1];  # plan action

                    if (visited_node.has_key(child_state) == False):
                        loc_stack.push(child_state)  #
                        parent_child_map[child_state] = [parent_node, child_action]  # add child state
                    temp = temp + 1


def uniformCostSearch(problem):
    loc_pqueue = util.PriorityQueue()
    visited_node = {}
    parent_child_map = {}
    path_cost = 0

    start_node = problem.getStartState()
    parent_child_map[start_node] = []
    loc_pqueue.push(start_node, path_cost)


    while (loc_pqueue.isEmpty() == False):

        parent_node = loc_pqueue.pop()

        if (parent_node != problem.getStartState()):
            path_cost = parent_child_map[parent_node][2]

        if (problem.isGoalState(parent_node)):
            pathlist = traverse_path(parent_node, parent_child_map,3)
            pathlist.reverse()
            return pathlist

        elif (visited_node.has_key(parent_node) == False):
            visited_node[parent_node] = []
            sucessor_list = problem.getSuccessors(parent_node)
            no_of_child = len(sucessor_list)
            if (no_of_child > 0):
                temp = 0
                while (temp < no_of_child):
                    child_nodes = sucessor_list[temp]
                    child_state = child_nodes[0];
                    child_action = child_nodes[1];
                    child_cost = child_nodes[2];
                    gvalue = path_cost + child_cost
                    if (visited_node.has_key(child_state) == False):
                        loc_pqueue.push(child_state, gvalue)
                    if (parent_child_map.has_key(child_state) == False):
                        parent_child_map[child_state] = [parent_node, child_action, gvalue]
                    else:
                        if (child_state != start_node):
                            stored_cost = parent_child_map[child_state][2]
                            if (stored_cost > gvalue):
                                parent_child_map[child_state] = [parent_node, child_action, gvalue]
                    temp = temp + 1

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    loc_pqueue = util.PriorityQueue()
    visited_node = {}
    parent_child_map = {}
    direction_list = []
    path_cost = 0
    heuristic_value = 0

    start_node = problem.getStartState()
    parent_child_map[start_node] = []
    loc_pqueue.push(start_node, heuristic_value)



    while (loc_pqueue.isEmpty() == False):

        parent_node = loc_pqueue.pop()

        if (parent_node != problem.getStartState()):
            path_cost = parent_child_map[parent_node][2]

        if (problem.isGoalState(parent_node)):
            pathlist = traverse_path(parent_node, parent_child_map,4)
            pathlist.reverse()
            return pathlist

        elif (visited_node.has_key(parent_node) == False):
            visited_node[parent_node] = []
            sucessor_list = problem.getSuccessors(parent_node)
            no_of_child = len(sucessor_list)
            if (no_of_child > 0):
                temp = 0
                while (temp < no_of_child):
                    child_nodes = sucessor_list[temp]
                    child_state = child_nodes[0];
                    child_action = child_nodes[1];
                    child_cost = child_nodes[2];

                    heuristic_value = heuristic(child_state, problem)
                    gvalue = path_cost + child_cost
                    fvalue = gvalue + heuristic_value

                    if (visited_node.has_key(child_state) == False):
                        loc_pqueue.push(child_state, fvalue)
                    if (parent_child_map.has_key(child_state) == False):
                        parent_child_map[child_state] = [parent_node, child_action, gvalue, fvalue]
                    else:
                        if (child_state != start_node):
                            stored_fvalue = parent_child_map[child_state][3]
                            if (stored_fvalue > fvalue):
                                parent_child_map[child_state] = [parent_node, child_action, gvalue, fvalue]
                    temp = temp + 1


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
