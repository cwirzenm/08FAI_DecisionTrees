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
        self.mapping = {'parent': [], 'condition': [], 'name': []} # todo improve this
        self.stack = set([])
        self.dataToConditions = {}
        self.label = None

    def add_root(self, name):
        self.mapping['parent'].append(None)
        self.mapping['condition'].append(None)
        self.mapping['name'].append(name)
        self.push_stack(name)

    def add_node(self, parent, condition, name, isGoalNode=False):
        if parent not in self.mapping['name']: raise Exception(f"Parent node {parent} not found")
        if condition not in self.dataToConditions[parent]: raise Exception(f"{parent} does not contain {condition}")
        self.mapping['parent'].append(parent)
        self.mapping['condition'].append(condition)
        self.mapping['name'].append(name)
        if not isGoalNode: self.push_stack(name)

    def push_stack(self, name):
        for condition in self.dataToConditions[name]: self.stack.add((name, condition))

    def is_finished(self) -> bool: return len(self.stack) == 0

    def get_lowest_gi(self, df: pd.DataFrame) -> str:
        return df.columns[np.argmin([gini_index(df.loc[:, [col, self.label]]) for col in df.iloc[:, :-1]])]

    def traverse_up(self, parent, condition) -> list:
        to_drop = []
        root_traversal, condition_traversal = parent, condition
        while root_traversal is not None:
            to_drop.append((root_traversal, condition_traversal))
            root_i = self.mapping['name'].index(root_traversal)
            root_traversal = self.mapping['parent'][root_i]
            condition_traversal = self.mapping['condition'][root_i]
        return to_drop

    def train(self, dataset: pd.DataFrame):
        # init class variables
        self.dataToConditions = {col: dataset.loc[:, col].unique() for col in dataset}
        self.label = dataset.columns[-1]

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

            decisions = df[self.label].unique()
            if len(decisions) == 1:
                self.add_node(parent, condition, decisions[0], isGoalNode=True)
                continue

            # calculate the gini indices and find the lowest one
            lowest_gi = self.get_lowest_gi(df)

            # create node
            self.add_node(parent, condition, lowest_gi)

        print('Decision Tree completed.', *((key, val) for key, val in self.mapping.items()), sep='\n', end='\n\n')

    def print(self):
        # pretty print the tree
        return


if __name__ == '__main__':
    weather_data = pd.read_csv('weather-data.csv')
    weather_data = weather_data.drop(['Day'], axis=1)  # dropping 'Day' column because it's not needed
    dt = DT()
    dt.train(weather_data)

    car_data = pd.read_csv('car_data.csv')
    dt = DT()
    dt.train(car_data)
