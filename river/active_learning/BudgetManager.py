class BudgetManager():
    def __init__(self) -> None:
        pass

    def isAbove(value):
        pass

    def getLabelAcqReport():
        pass


class FixedBM(BudgetManager):
    def __init__(self, budgetValue: float) -> None:
        self.budget = budgetValue
        self.acquiredLabels = 0

    
    def isAbove(self, value: float) -> bool:
        acquireLabel = False
        if (value >= 1.0 - self.budget):
            acquireLabel = True
        
        if (acquireLabel):
            self.acquiredLabels += 1
        
        return acquireLabel
    
    
    def getLabelAcqReport(self) -> int:
        aux = self.acquiredLabels
        self.acquiredLabels = 0
        return aux
        