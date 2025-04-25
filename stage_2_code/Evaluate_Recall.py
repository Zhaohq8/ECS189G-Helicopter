
"""
Concrete evaluator for micro-averaged recall.
"""
from local_code.base_class.evaluate import evaluate
from sklearn.metrics import recall_score


class Evaluate_Recall(evaluate):
    data = None

    def evaluate(self):
        print('evaluating recall...')
        return recall_score(
            self.data['true_y'],
            self.data['pred_y'],
            average='micro'
        )