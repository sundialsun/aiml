# aiml
9
 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
model = LinearRegression()
model.fit(X, y)
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_new, y_pred, color='red', linewidth=3, label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


8

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
y = [0, 0, 0, 1, 1, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
new_data_point = [[4, 4]]
predicted_class = knn_classifier.predict(new_data_point)
print(f'Predicted class for {new_data_point}: {predicted_class}')

6


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
np.random.seed(42)
data = np.random.rand(100, 20)
labels = np.random.choice([0, 1], size=100)
data[0, 0] = np.nan
data[1, 1] = np.nan
data[2, 2] = np.nan
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(20)])
df['label'] = labels
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
train_predictions = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy:.2f}")
test_predictions = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Testing Accuracy: {test_accuracy:.2f}")


3

3 import csv
with open("trainingexamples.csv") as f:
csv_file = csv.reader(f)
data = list(csv_file)
specific = data[0][:-1]
general = [['?' for _ in range(len(specific))]
for _ in range(len(specific))]
for i in data:
target_value = i[-1]
for j in range(len(specific)):
if i[j] != specific[j]:
if target_value == "Yes":
specific[j] = "?"
general[j][j] = "?"
elif target_value == "No":
general[j][j] = specific[j]
else:general[j][j] = "?"
print("\nStep{}of".format(data.index(i)+1))
print(specific)
print(general)
gh = [i for i in general if any(val != '?' for val in i)]
print("\nFinalSpecifichypothesis:\n", specific)
 print("\nFinalGeneralhypothesis:\n", gh)




5


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def train_neural_network(X, y, epochs=1000, learning_rate=0.1):
    input_layer_neurons = X.shape[1]
    output_neurons = 1

    weights_hidden = np.random.uniform(size=(input_layer_neurons, 8))
    weights_output = np.random.uniform(size=(8, output_neurons))

    for epoch in range(epochs):
        # Forward Propagation
        hidden_layer_input = np.dot(X, weights_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_output)
        predicted_output = sigmoid(output_layer_input)

        # Backpropagation
        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(weights_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights
        weights_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_hidden += X.T.dot(d_hidden_layer) * learning_rate

    return weights_hidden, weights_output

def predict(X, weights_hidden, weights_output):
    hidden_layer_input = np.dot(X, weights_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_output)
    predicted_output = sigmoid(output_layer_input)

    return predicted_output

# Input and Output
input_data = np.array([[0.66666667, 1.],
                       [0.33333333, 0.55555556],
                       [1., 0.66666667]])

actual_output = np.array([[0.92], [0.86], [0.89]])

# Train the Neural Network
X_train, X_test, y_train, y_test = train_test_split(input_data, actual_output, test_size=0.2, random_state=42)

weights_hidden, weights_output = train_neural_network(X_train, y_train)

# Predictions
predicted_output = predict(X_test, weights_hidden, weights_output)

# Evaluate and Print Results
mse = mean_squared_error(y_test, predicted_output)

print("Actual Output:")
print(y_test)

print("\nPredicted Output:")
print(predicted_output)

print("\nMean Squared Error: {:.4f}".format(mse))

4

class Node:
    def __init__(self, attribute=None, branches=None, label=None):
        self.attribute = attribute
        self.branches = branches if branches is not None else {}
        self.label = label

def id3(data, target_attribute, attributes):
    # If all examples have the same label, return a leaf node with that label
    if len(set(data[target_attribute])) == 1:
        return Node(label=data[target_attribute].iloc[0])

    # If no attributes are left, return a leaf node with the majority label
    if len(attributes) == 0:
        majority_label = data[target_attribute].mode().iloc[0]
        return Node(label=majority_label)

    # Choose the best attribute to split on
    best_attribute = choose_best_attribute(data, target_attribute, attributes)

    # Create a new internal node with the chosen attribute
    root = Node(attribute=best_attribute)

    # Recursively build the tree for each branch
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if len(subset) == 0:
            # If a branch is empty, create a leaf node with the majority label
            majority_label = data[target_attribute].mode().iloc[0]
            root.branches[value] = Node(label=majority_label)
        else:
            # Recursively build the tree for the branch
            root.branches[value] = id3(subset, target_attribute, [attr for attr in attributes if attr != best_attribute])

    return root

def choose_best_attribute(data, target_attribute, attributes):
    # Calculate the information gain for each attribute
    info_gain = {attr: calculate_information_gain(data, attr, target_attribute) for attr in attributes}

    # Choose the attribute with the highest information gain
    best_attribute = max(info_gain, key=info_gain.get)
    return best_attribute

def calculate_information_gain(data, attribute, target_attribute):
    # Calculate entropy before the split
    entropy_before = calculate_entropy(data[target_attribute])

    # Calculate weighted average entropy after the split
    values = data[attribute].unique()
    weighted_entropy_after = sum((len(data[data[attribute] == value]) / len(data)) * calculate_entropy(data[data[attribute] == value][target_attribute]) for value in values)

    # Calculate information gain
    info_gain = entropy_before - weighted_entropy_after
    return info_gain

def calculate_entropy(labels):
    # Calculate entropy for a set of labels
    entropy = -sum((labels.value_counts() / len(labels)) * np.log2(labels.value_counts() / len(labels)))
    return entropy

# Example usage with the provided dataset
import pandas as pd
import numpy as np

# Create the dataset
data = pd.DataFrame({
    'wind': ['strong', 'weak', 'weak', 'strong', 'strong'],
    'sunny': ['no', 'yes', 'yes', 'no', 'yes'],
    'humidity': ['high', 'normal', 'high', 'high', 'normal']
})

# Add the target attribute
data['play'] = ['no', 'yes', 'yes', 'no', 'yes']

# Define the attributes
attributes = ['wind', 'sunny', 'humidity']

# Build the ID3 tree
tree = id3(data, 'play', attributes)

# Function to print the tree
def print_tree(node, depth=0):
    if node.label is not None:
        print("  " * depth + "Label: " + str(node.label))
    else:
        print("  " * depth + "Attribute: " + str(node.attribute))
        for value, branch in node.branches.items():
            print("  " * (depth + 1) + "Value: " + str(value))
            print_tree(branch, depth + 2)

# Print the resulting tree
print_tree(tree)




1

def aStarAlgo(start_node, stop_node):
    

    open_set = set(start_node) # {A}, len{open_set}=1
    closed_set = set()
    g = {} # store the distance from starting node
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node # parents['A']='A"

    while len(open_set) > 0 :
        n = None
        
        for v in open_set: # v='B'/'F'
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v # n='A'

        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbors(n):
             # nodes 'm' not in first and last set are added to first
             # n is set its parent
                if m not in open_set and m not in closed_set:
                    open_set.add(m)      # m=B weight=6 {'F','B','A'} len{open_set}=2
                    parents[m] = n       # parents={'A':A,'B':A} len{parent}=2
                    g[m] = g[n] + weight # g={'A':0,'B':6, 'F':3} len{g}=2


            #for each node m,compare its distance from start i.e g(m) to the 
            #from start through n node
                else:
                    if g[m] > g[n] + weight:
                    #update g(m)
                        g[m] = g[n] + weight
                    #change parent of m to n
                        parents[m] = n

                    #if m in closed set,remove and add to open
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)

        if n == None:
            print('Path does not exist!')
            return None

        # if the current node is the stop_node
        # then we begin reconstructin the path from it to the start_node
        if n == stop_node:
            path = []

            while parents[n] != n:
                path.append(n)
                n = parents[n]

            path.append(start_node)

            path.reverse()

            print('Path found: {}'.format(path))
            return path


        # remove n from the open_list, and add it to closed_list
        # because all of his neighbors were inspected
        open_set.remove(n)# {'F','B'} len=2
        closed_set.add(n) #{A} len=1

    print('Path does not exist!')
    return None

#define fuction to return neighbor and its distance
#from the passed node
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
#for simplicity we ll consider heuristic distances given
#and this function returns heuristic distance for all nodes
 
def heuristic(n):
    H_dist = {
        'A': 10,
        'B': 8,
        'C': 5,
        'D': 7,
        'E': 3,
        'F': 6,
        'G': 5,
        'H': 3,
        'I': 1,
        'J': 0
    }

    return H_dist[n]

#Describe your graph here
Graph_nodes = {
    
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1),('H', 7)] ,
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)],

}
aStarAlgo('A', 'J')




2



class Graph:
    def __init__(self, graph, heuristicNodeList, startNode):
  #instantiate graph object with graph topology, heuristic values, start node
        
        self.graph = graph
        self.H=heuristicNodeList
        self.start=startNode
        self.parent={}
        self.status={}
        self.solutionGraph={}
     
    def applyAOStar(self):        
 # starts a recursive AO* algorithm
        self.aoStar(self.start, False)
 
    def getNeighbors(self, v):     
# gets the Neighbors of a given node
        return self.graph.get(v,'')
    
    def getStatus(self,v):         
# return the status of a given node
        return self.status.get(v,0)
    
    def setStatus(self,v, val):   
 # set the status of a given node
        self.status[v]=val
    
    def getHeuristicNodeValue(self, n):
        return self.H.get(n,0)     
# always return the heuristic value of a given node
 
    def setHeuristicNodeValue(self, n, value):
        self.H[n]=value            
# set the revised heuristic value of a given node 
        
    
    def printSolution(self):
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:",self.start)
        print("------------------------------------------------------------")
        print(self.solutionGraph)
        print("------------------------------------------------------------")
    
    def computeMinimumCostChildNodes(self, v):  
# Computes the Minimum Cost of child nodes of a given node v     
        minimumCost=0
        costToChildNodeListDict={}
        costToChildNodeListDict[minimumCost]=[]
        flag=True
        for nodeInfoTupleList in self.getNeighbors(v):  
# iterate over all the set of child node/s
            cost=0
            nodeList=[]
            for c, weight in nodeInfoTupleList:
                cost=cost+self.getHeuristicNodeValue(c)+weight
                nodeList.append(c)
            
            if flag==True:  
# initialize Minimum Cost with the cost of first set of child node/s 
                minimumCost=cost
                costToChildNodeListDict[minimumCost]=nodeList      
# set the Minimum Cost child node/s
                flag=False
            else:                              
 # checking the Minimum Cost nodes with the current Minimum Cost   
                if minimumCost>cost:
                    minimumCost=cost
                    costToChildNodeListDict[minimumCost]=nodeList  
# set the Minimum Cost child node/s
                
              
        return minimumCost, costToChildNodeListDict[minimumCost]  
 # return Minimum Cost and Minimum Cost child node/s
 
                     
    
    def aoStar(self, v, backTracking):     
# AO* algorithm for a start node and backTracking status flag
        
        print("HEURISTIC VALUES  :", self.H)
        print("SOLUTION GRAPH    :", self.solutionGraph)
        print("PROCESSING NODE   :", v)
        print("--------------------------------------------------------------------")
        
        if self.getStatus(v) >= 0:       
 # if status node v >= 0, compute Minimum Cost nodes of v
            minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
            self.setHeuristicNodeValue(v, minimumCost)
            self.setStatus(v,len(childNodeList))
            
            solved=True                   
# check the Minimum Cost nodes of v are solved   
            for childNode in childNodeList:
                self.parent[childNode]=v
                if self.getStatus(childNode)!=-1:
                    solved=solved & False
            
            if solved==True:            
 # if the Minimum Cost nodes of v are solved, set the current node status as solved(-1)
                self.setStatus(v,-1)    
                self.solutionGraph[v]=childNodeList 
# update the solution graph with the solved nodes which may be a part of solution  
            
            
            if v!=self.start:          
 # check the current node is the start node for backtracking the current node value    
                self.aoStar(self.parent[v], True)   
# backtracking the current node value with backtracking status set to true
                
            if backTracking==False:    
 # check the current call is not for backtracking 
                for childNode in childNodeList:  
 # for each Minimum Cost child node
                    self.setStatus(childNode,0)   
# set the status of child node to 0(needs exploration)
                    self.aoStar(childNode, False)
 # Minimum Cost child node is further explored with backtracking status as false
                 
        
                                       
h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph1 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]   
}
G1= Graph(graph1, h1, 'A')
G1.applyAOStar() 
G1.printSolution()
 
h2 = {'A': 1, 'B': 6, 'C': 12, 'D': 10, 'E': 4, 'F': 4, 'G': 5, 'H': 7}  # Heuristic values of Nodes 
graph2 = {                                        # Graph of Nodes and Edges 
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],      # Neighbors of Node 'A', B, C & D with repective weights 
    'B': [[('G', 1)], [('H', 1)]],                # Neighbors are included in a list of lists
    'D': [[('E', 1), ('F', 1)]]                   # Each sublist indicate a "OR" node or "AND" nodes
}
 
G2 = Graph(graph2, h2, 'A')                       # Instantiate Graph object with graph, heuristic values and start Node
G2.applyAOStar()                                  # Run the AO* algorithm
G2.printSolution()                

