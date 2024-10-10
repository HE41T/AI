import numpy as np
from neuralnetwork_np_round import NeuralNetwork

if __name__== "__main__":

    neural_network = NeuralNetwork()
    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)
    training_inputs = np.array([[1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7],
                                [1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6],
                                [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5],
                                [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4],
                                [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3],
                                [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2],
                                [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1]])
    training_outputs = np.array([[  2, 2, 2, 2, 2, 2, 2,
                                    2, 1, 1, 1, 1, 1, 2,
                                    2, 1, 0, 0, 0, 1, 2,
                                    2, 1, 0, 0, 0, 1, 2,
                                    2, 1, 0, 0, 0, 1, 2,
                                    2, 1, 1, 1, 1, 1, 2,
                                    2, 2, 2, 2, 2, 2, 2]])

    neural_network.train(training_inputs, training_outputs, 35000)
    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)
    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))

    print("Considering New Situation:", user_input_one, user_input_two)
    print("New Output data: ")
    print(np.round(neural_network.think(np.array([user_input_one, user_input_two]))))
    print("W O W we did it !!!")
    # ทำให้ทศนิยมปัดขึ้น


