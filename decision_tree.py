import numpy as np
import pandas as pd


def gini_impurity(probability: np.ndarray) -> float:
    result = 1
    for p in probability: result -= p ** 2
    return result


def gini_index(data: pd.DataFrame) -> float:
    data = data.value_counts()
    conditions = set(x[0][0] for x in data.items())

    weighted_gini_impurities = []
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
        self.stack = set()
        self.dataToConditions = {}
        self.label = None
        self.decisions = None

    def add_root(self, name):
        self.mapping.append(('Root', 'Decision Tree', name))
        self.push_stack(name)

    def add_node(self, parent, condition, name, isGoalNode=False):
        if parent not in [mapping[2] for mapping in self.mapping]: raise Exception(f"Parent node {parent} not found")
        if condition not in self.dataToConditions[parent]: raise Exception(f"{parent} does not contain {condition}")
        self.mapping.append((parent, condition, name))
        if not isGoalNode: self.push_stack(name)

    def push_stack(self, name):
        for condition in self.dataToConditions[name]: self.stack.add((name, condition))

    def is_finished(self) -> bool: return len(self.stack) == 0

    def get_lowest_gi(self, df: pd.DataFrame) -> str:
        return df.columns[np.argmin([gini_index(df.loc[:, [col, self.label]]) for col in df.iloc[:, :-1]])]

    def traverse_up(self, parent, condition) -> list:
        to_drop = []
        root_traversal, condition_traversal = parent, condition
        while root_traversal != 'Root':
            to_drop.append((root_traversal, condition_traversal))
            root_i = [mapping[2] for mapping in self.mapping].index(root_traversal)
            root_traversal = self.mapping[root_i][0]
            condition_traversal = self.mapping[root_i][1]
        return to_drop

    def create(self, dataset: pd.DataFrame):
        # init class variables
        self.dataToConditions = {col: dataset.loc[:, col].unique() for col in dataset}
        self.label = dataset.columns[-1]
        self.decisions = dataset[self.label].unique()

        # calculate the gini indices and find the lowest one
        lowest_gi = self.get_lowest_gi(dataset)

        # create root node
        self.add_root(lowest_gi)

        # create the rest of the tree
        while not self.is_finished():
            # process the top item from the stack
            parent, condition = self.stack.pop()

            # filter and drop the upstream nodes
            df = dataset
            for col, row in self.traverse_up(parent, condition):
                df = df.loc[dataset[col] == row].drop(col, axis=1)

            decisions_left = df[self.label].unique()
            if len(decisions_left) == 1:
                self.add_node(parent, condition, decisions_left[0], isGoalNode=True)
                continue

            # calculate the gini indices and find the lowest one
            lowest_gi = self.get_lowest_gi(df)

            # create node
            self.add_node(parent, condition, lowest_gi)

        print('Decision Tree completed.')
        print(f"{len(set(self.mapping))} unique nodes in total")
        print(*(mapping for mapping in self.mapping), sep='\n', end='\n\n')

    def print_tree(self):
        def create_tree(parent='Root', level=0) -> str:
            string = str()
            children = [mapping for mapping in self.mapping if mapping[0] == parent]
            for mapping in children:
                string += "\t" * level + f"{mapping[0]} is {mapping[1]} -> {mapping[2]}" + '\n'
                string += create_tree(mapping[2], level + 1)
            return string

        print(create_tree())

    def decision(self, branches: dict):
        if not branches:
            # todo full distribution
            return
        branches_left = branches
        iter_resolutions = {}
        for branch in branches:
            if len(branches_left) > 1:
                # todo accommodate multiple root nodes
                #  accumulate resolutions
                #  return current

                # remove from iter_branches
                return
            elif len(branches_left) > 0:

                # remove from iter_branches
                return


if __name__ == '__main__':
    weather_data = pd.read_csv('weather-data.csv')
    weather_data = weather_data.drop(['Day'], axis=1)  # dropping 'Day' column because it's not needed
    dt = DT()
    dt.create(weather_data)
    dt.print_tree()
    # print(dt.decision({
    #         'Outlook': 'Overcast',
    #         'Temperature': 'Cool',
    #         'Humidity': 'Normal',
    #         'Wind': 'Strong'
    # }))  # return decision
    # print(dt.decision({
    #         'Outlook': 'Overcast',
    #         'Temperature': 'Cool',
    #         'Humidity': 'Normal'
    # }))  # return decision
    # print(dt.decision({
    #         'Outlook': 'Overcast',
    #         'Temperature': 'Cool'
    # }))  # return decision distribution
    # print(dt.decision({
    #         'Outlook': 'Overcast'
    # }))  # return decision distribution
    print(dt.decision({}))  # return decision distribution

    car_data = pd.read_csv('car_data.csv')
    dt = DT()
    dt.create(car_data)
    dt.print_tree()
    # dt.print_summary()
