from src.estimator import NeuralNetwork
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # make moons
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42, shuffle=True)
    nn = NeuralNetwork(units_per_layer=[])
    clf = nn.fit(X, y)

    # plot decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(clf, X)
    plt.show()