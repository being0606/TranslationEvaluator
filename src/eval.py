import evaluate
import pandas as pd
import os
from tqdm import tqdm  # tqdm 임포트 추가
from time import perf_counter

class TranslationEvaluator:
    def __init__(self, predictions, references, output_path='evaluation_results.csv'):
        self.predictions = predictions
        self.references = references
        self.output_path = output_path
        self._validate_inputs()
        self.metrics = {
            'bleu': evaluate.load('bleu'),
            'meteor': evaluate.load('meteor'),
            # 'ter': evaluate.load('ter'),
            'rouge': evaluate.load('rouge')
        }
        self.results = []

    def _validate_inputs(self):
        if not isinstance(self.predictions, list) or not isinstance(self.references, list):
            raise ValueError("Predictions and references should be lists.")
        if len(self.predictions) != len(self.references):
            raise ValueError("Predictions and references must have the same length.")

    def compute_scores(self):
        # tqdm을 사용하여 진행 상황 시각화
        for idx, (pred, ref) in enumerate(tqdm(zip(self.predictions, self.references), total=len(self.predictions), desc="Evaluating")):
            # ref가 문자열이면 리스트의 리스트로 변환
            if isinstance(ref, str):
                ref_for_metrics = [ref]
                sentence_references = ref
            elif isinstance(ref, list):
                ref_for_metrics = ref
                sentence_references = ' ||| '.join(ref)
            else:
                raise ValueError("Reference must be a string or a list of strings.")

            row = {
                'index': idx,
                'sentence_references': sentence_references,
                'sentence_predictions': pred
            }

            for metric_name, metric in self.metrics.items():
                t = perf_counter()
                try:
                    if metric_name == 'bleu':
                        # BLEU는 references를 리스트의 리스트로 받음
                        score = metric.compute(predictions=[pred], references=[ref_for_metrics])
                        row[f'Metric_{metric_name}'] = score['bleu']
                    elif metric_name == 'rouge':
                        # ROUGE는 references를 문자열의 리스트로 받음
                        score = metric.compute(predictions=[pred], references=[ref])
                        row.update({f'Metric_{metric_name}_{k}': v for k, v in score.items()})
                    else:
                        # 다른 메트릭들도 적절히 처리
                        score = metric.compute(predictions=[pred], references=[ref])
                        # if metric_name == 'ter':
                        #     row[f'Metric_{metric_name}'] = score['score']
                        # else:
                        #     row[f'Metric_{metric_name}'] = score[metric_name]
                except Exception as e:
                    print(f"Error computing {metric_name} for index {idx}: {e}")
                    row[f'Metric_{metric_name}'] = None
                    
                print(f"{metric_name} took {perf_counter() - t:.2f} seconds")

            self.results.append(row)

    def save_to_csv(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_path, index=False)
        print(f"Results saved to {os.path.abspath(self.output_path)}")

# 메인 함수로 실행할 경우
if __name__ == "__main__":
    df_sentence_pair = pd.read_csv('../data/processed/batchoutput_1000_gpt-4o.csv')
    df_sentence_pair = df_sentence_pair[:10] # 일단 10개만 테스트

    # 예측 및 참조 문장 리스트 생성
    predictions = df_sentence_pair["english"].tolist()
    references = df_sentence_pair["predictions_Eng"].tolist()

    # references를 리스트의 리스트로 변환
    references = [[ref] if isinstance(ref, str) else ref for ref in references]

    evaluator = TranslationEvaluator(predictions, references)
    evaluator.compute_scores()
    # evaluator.save_to_csv()