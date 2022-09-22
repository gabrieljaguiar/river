from river.active_learning.ActiveLearning import ALRandom
from river.active_learning.FixedBM import FixedBM
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.datasets import synth


dataset = synth.Agrawal(
    classification_function=0,
    seed=42
)

base_classifier = HoeffdingAdaptiveTreeClassifier()
budgetManager = FixedBM(0.5)
al_classifier = ALRandom(base_classifier, budgetManager)

it = 0

for x, y in dataset.take(10000):
    al_classifier.learn_one(x, y)
    
    it +=1
    if (it % 500 == 0):
        labedledInstances = al_classifier.getLastLabelRebort()
        print ("{} were labeled in the last window, it is {}\% of the instances".format(
            labedledInstances, labedledInstances/500.0
        ))
        print (al_classifier.predict_proba_one(x))
