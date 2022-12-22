import math
import numpy as np
import pandas as pd
from utils import Timer


def gini_impurity(probability: np.ndarray) -> float:
    """Calculate the gini impurity"""
    result = 1
    for p in probability: result -= p ** 2
    return result


def gini_index(data: pd.DataFrame) -> float:
    """Calculate the gini index"""
    data = data.value_counts()
    conditions = set(x[0][0] for x in data.items())

    weighted_gini_impurities = list()
    weights = []
    for c in conditions:
        occurrences = data.loc[c].sum()
        gi = gini_impurity(data.loc[c].values / occurrences)
        weighted_gini_impurities.append(gi * occurrences)
        weights.append(occurrences)
    return sum(weighted_gini_impurities / sum(weights))


class DT:
    def __init__(self):
        self.mapping = list()
        self.levels = list()
        self.weights = list()
        self.stack = list()
        self.dataToConditions = {}
        self.label = None
        self.decisions = None

    def add_node(self, parent: str, condition: str, name: str, level: int, weight: int, isGoalNode=False):
        """Add node to the tree"""
        if parent not in [mapping[2] for mapping in self.mapping]: raise Exception(f"Parent node {parent} not found")
        if not isGoalNode and condition not in self.dataToConditions[parent]: raise Exception(f"{parent} does not contain {condition}")
        self.mapping.append((parent, condition, name))
        self.levels.append(level)
        self.weights.append(weight)
        if not isGoalNode: self.push_stack(name)

    def push_stack(self, name):
        """Push onto the processing stack"""
        for condition in self.dataToConditions[name]:
            if (name, condition) not in self.stack:
                self.stack.insert(0, (name, condition))

    def is_finished(self) -> bool: return len(self.stack) == 0

    def get_lowest_gi(self, df: pd.DataFrame) -> str:
        """Get the lowest gini index"""
        return df.columns[np.argmin([gini_index(df.loc[:, [col, self.label]]) for col in df.iloc[:, :-1]])]

    def traverse_up(self, parent, condition) -> list:
        """Read ancestors of the node"""
        to_drop = []
        root_traversal, condition_traversal = parent, condition
        while root_traversal != 'Root':
            to_drop.append((root_traversal, condition_traversal))
            root_i = [mapping[2] for mapping in self.mapping].index(root_traversal)
            root_traversal = self.mapping[root_i][0]
            condition_traversal = self.mapping[root_i][1]
        return to_drop

    def create(self, dataset: pd.DataFrame):
        """Create a decision tree from a dataset"""
        # init class variables
        self.label = dataset.columns[-1]
        self.decisions = dataset[self.label].unique()
        self.dataToConditions = {col: dataset.loc[:, col].unique() for col in dataset if col != self.label}

        # calculate the gini indices and find the lowest one
        lowest_gi = self.get_lowest_gi(dataset)

        # create root node
        self.mapping.append(('Root', 'Decision Tree', lowest_gi))
        self.levels.append(0)
        self.weights.append(math.prod(len(self.dataToConditions[x]) for x in self.dataToConditions))
        self.push_stack(lowest_gi)

        # create the rest of the tree
        while not self.is_finished():
            # process the top item from the stack
            parent, condition = self.stack.pop(0)

            # calculate the upstream nodes
            upstream = self.traverse_up(parent, condition)

            # level is the number of upstream nodes
            level = len(upstream)

            # weight is the number of all possible downstream nodes
            weight = math.prod(len(self.dataToConditions[d]) for d in self.dataToConditions if d not in [u[0] for u in upstream])

            # drop the upstream nodes
            df = dataset
            for col, row in upstream:
                df = df.loc[dataset[col] == row].drop(col, axis=1)

            # check all possible resolutions
            resolutions_left = df[self.label].unique()

            # if there is only one resolution possible, create a goal node
            if len(resolutions_left) == 1:
                self.add_node(parent, condition, resolutions_left[0], level, weight, isGoalNode=True)
                continue

            # calculate the gini indices and find the lowest one
            lowest_gi = self.get_lowest_gi(df)

            # create node
            self.add_node(parent, condition, lowest_gi, level, weight)

        print('\nDecision Tree completed.')
        print(f"{len(set(self.mapping))} unique nodes in total")
        print(*(mapping for mapping in self.mapping), sep='\n', end='\n\n')

    def print_tree(self):
        """Pretty print the decision tree"""
        print(*("\t" * level + f"{mapping[0]} is {mapping[1]} -> {mapping[2]}" + '\n'
                for mapping, level in zip(self.mapping, self.levels)))

    def print_decision(self, choices: dict):
        """Make decisions based on input"""

        # check if choices are valid
        for name, choice in choices.items():
            if name not in self.dataToConditions.keys() or choice not in self.dataToConditions[name]:
                raise Exception('Invalid choice.')

        # get all possible resolutions of the tree
        resolutions = list(zip(self.mapping, self.levels, self.weights))

        # iterate through the mapping removing the unnecessary resolutions
        removeAllChildren = False
        for mapping, level, weight in zip(self.mapping, self.levels, self.weights):
            parent = mapping[0]
            condition = mapping[1]

            # if we're not removing the node then do nothing
            if not removeAllChildren: pass
            # if all children of the previous node have been removed
            elif removeAllChildren == level: removeAllChildren = False
            # remove the child node
            else: resolutions.remove((mapping, level, weight)); continue

            # if node is not the choice, remove it with all it's children
            if choices.get(parent) and choices.get(parent) != condition:
                # remove this and all children from possible resolutions
                resolutions.remove((mapping, level, weight))
                removeAllChildren = level

        # dict of all possible decisions
        result = dict((decision, 0) for decision in self.decisions)

        # generate a distribution out of all remaining resolutions
        goal_resolutions = [(r[0][2], r[2]) for r in resolutions if r[0][2] in self.decisions]
        for goal, weight in goal_resolutions: result[goal] += weight
        total = sum(result.values())
        for resolution in result: result[resolution] /= total

        # print the result
        print(f"Choices {choices} ->", *(f"\t{goal} = {p:5f}" for goal, p in result.items() if p), sep='\n', end='\n\n')


if __name__ == '__main__':
    with Timer():
        weather_data = pd.read_csv('weather-data.csv')
        weather_data = weather_data.drop(['Day'], axis=1)  # dropping 'Day' column because it's not needed
        dt = DT()
        dt.create(weather_data)
        dt.print_tree()
        dt.print_decision({})  # return decision distribution

    # dt.print_decision({
    #     'Wind': 'Strong'
    # })  # return decision
    # dt.print_decision({
    #     'Temperature': 'Cool',
    #     'Humidity': 'Normal'
    # })  # return decision
    # dt.print_decision({
    #     'Outlook': 'Overcast',
    #     'Temperature': 'Cool'
    # })  # return decision distribution
    # dt.print_decision({
    #     'Temperature': 'Cool'
    # })  # return decision distribution

    with Timer():
        car_data = pd.read_csv('car_data.csv')
        dt = DT()
        dt.create(car_data)
        dt.print_tree()
        dt.print_decision({})
