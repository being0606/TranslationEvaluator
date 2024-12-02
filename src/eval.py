import evaluate
import pandas as pd
import os

class TranslationEvaluator:
    def __init__(self, predictions, references, output_path='evaluation_results.csv'):
        self.predictions = predictions
        self.references = references
        self.output_path = output_path
        self._validate_inputs()
        self.metrics = {
            'bleu': evaluate.load('bleu'),
            'meteor': evaluate.load('meteor'),
            'ter': evaluate.load('ter'),
            'rouge': evaluate.load('rouge')
        }
        self.results = []

    def _validate_inputs(self):
        if not isinstance(self.predictions, list) or not isinstance(self.references, list):
            raise ValueError("Predictions and references should be lists.")
        if len(self.predictions) != len(self.references):
            raise ValueError("Predictions and references must have the same length.")
        ref_len = len(self.references[0])
        for ref in self.references:
            if len(ref) != ref_len:
                raise ValueError("Each reference list must have the same number of sentences.")

    def compute_scores(self):
        for idx, (pred, ref) in enumerate(zip(self.predictions, self.references)):
            row = {
                'index': idx,
                'sentence_references': ' ||| '.join(ref),
                'sentence_predictions': pred
            }
            for metric_name, metric in self.metrics.items():
                score = metric.compute(predictions=[pred], references=[ref])
                if metric_name == 'rouge':
                    row.update({f'Metric_{metric_name}_{k}': v for k, v in score.items()})
                elif metric_name == 'ter':
                    row[f'Metric_{metric_name}'] = score['score']
                else:
                    row[f'Metric_{metric_name}'] = score[metric_name]
            self.results.append(row)

    def save_to_csv(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_path, index=False)
        print(f"Results saved to {os.path.abspath(self.output_path)}")

# 메인함수로 실행할경우
if __name__ == "__main__":
    references = [
        # ["이것은 예시 번역입니다.", "이것은 또 다른 예시입니다."],
        # ["이것은 또 다른 예시입니다.", "또 다른 예시 번역입니다."]
        ["이것은 예시 번역입니다."],
        ["이것은 또 다른 예시입니다."]
    ]


    predictions = [
        "이것은 예시 번역입니다.",
        "또 다른 예시 번역입니다."
    ]

    evaluator = TranslationEvaluator(predictions, references)
    evaluator.compute_scores()
    evaluator.save_to_csv()