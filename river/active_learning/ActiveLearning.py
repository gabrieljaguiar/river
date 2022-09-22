from river import base
from random import Random
from .BudgetManager import BudgetManager, FixedBM


class ALRandom(base.Classifier):
    def __init__(
        self,
        classifier: base.Classifier,
        budgetManager: BudgetManager = FixedBM(0.1),
        seed: int = 42,
    ) -> None:
        self.budgetManager = budgetManager
        self.seed = seed
        self.random = Random(seed)
        self.classifier = classifier

    def learn_one(self, x, y, *args, **kwargs):
        value = self.random.uniform(0, 1)
        if self.budgetManager.isAbove(value):
            if y is not None:
                self.classifier.learn_one(x, y, *args, **kwargs)
            else:
                # Add instance to be labeled
                # After labeled call self.classifier.learn_one
                raise NotImplemented()

    def predict_proba_one(self, *args, **kwargs):
        return self.classifier.predict_proba_one(*args, **kwargs)

    def predict_one(self, *args, **kwargs):
        return self.classifier.predict_one(*args, **kwargs)

    def getLastLabelRebort(self) -> int:
        return self.budgetManager.getLabelAcqReport()


class ALUncertainty(base.Classifier):
    def __init__(
        self,
        classifier: base.Classifier,
        budget: float = 0.1,
        uncertaintyStrategy: str = "FixedUncertainty",
        fixedThreshold: float = 0.9,
        stepValue: float = 0.01,
        mercyPeriod: int = 100,
        seed: int = 42,
    ):
        self.classifier = classifier
        self.mercyPeriod = mercyPeriod
        self.costLabeling = 0
        self.iteration = 0
        self.budget = budget
        self.acquiredLabels = 0

        self.fixedThreshold = fixedThreshold

        self.varThreshold = 1.0

        self.stepValue = stepValue

        self.random = Random(seed)

        self.numClasses = -1

        self.strategyOptions = {
            "FixedUncertainty": self.fixedUncertainty,
            "VarUncertainty": self.varUncertainty,
            "RandVarUncertainty": self.randVarUncertainty,
            "SelSampling": self.selSampling,
        }

        assert (
            uncertaintyStrategy in self.strategyOptions.items(),
            "Strategy {} not available".format(uncertaintyStrategy),
        )

        self.uncertaintyFunction = self.strategyOptions[uncertaintyStrategy]

    def fixedUncertainty(self, incomingPosterior: float) -> bool:
        if incomingPosterior <= self.fixedThreshold:
            self.costLabeling += 1
            self.acquiredLabels += 1
            return True
        return False

    def varUncertainty(self, incomingPosterior):
        if incomingPosterior <= self.varThreshold:
            self.costLabeling += 1
            self.acquiredLabels += 1
            self.varThreshold *= 1.0 - self.stepValue
            return True
        else:
            self.varThreshold *= 1.0 + self.stepValue
            return False

    def randVarUncertainty(self, incomingPosterior):
        incomingPosterior /= self.random.gauss(0, 1) + 1.0
        return self.varUncertainty(incomingPosterior)

    def selSampling(self, incomingPosterior):
        p = abs(incomingPosterior - 1.0) / self.numClasses
        localBudget = self.budget / (self.budget + p)
        if self.random.uniform(0, 1) < localBudget:
            self.costLabeling += 1
            self.acquiredLabels += 1
            return True

    def getMaxPosterior(self, votes):
        outPosterior = 0.0
        if len(votes) > 1:
            outPosterior = max(votes.items())[1]

        return outPosterior

    def learn_one(self, x, y):
        self.iteration += 1
        if self.iteration <= self.mercyPeriod:
            self.classifier.learn_one(x, y)
            self.costLabeling += 1
        else:
            actualCost = (self.costLabeling - self.mercyPeriod) / (
                self.iteration - self.mercyPeriod
            )
            if actualCost <= self.budget:
                votes = self.classifier.predict_proba_one(x)
                if self.numClasses != len(votes.items()):
                    self.numClasses = len(votes.items())

                if self.uncertaintyFunction(self.getMaxPosterior(votes)):
                    self.classifier.learn_one(x, y)

    def getLastLabelRebort(self) -> int:
        aux = self.acquiredLabels
        self.acquiredLabels = 0
        return aux

    def predict_proba_one(self, *args, **kwargs):
        return self.classifier.predict_proba_one(*args, **kwargs)

    def predict_one(self, *args, **kwargs):
        return self.classifier.predict_one(*args, **kwargs)
