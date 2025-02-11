# euler-math

## 1. Evaluation
- 
```
bash scripts/run_eval.sh
```

## 2. Scoring
- `root_path(str)`: Default value is `results`.
- `datasets(list)`: Dataset list to grade the response of models. All datasets in directory will be evaluated if `None` was given.
- `languages(list)`: Language list to grade the response of models. All languages in directory will be evaluated if `None` was given.
```
bash scripts/run_score.sh
```

## 3. Language Consistency Score(LCS)
**Arguments**
- `root_path(str)`: Default value is `results`.
- `models(list)`: Model list to measure the LCS. All models in directory will be evaluated if `None` was given.
- `datasets(list)`: Dataset list to measure the LCS. All datasets in root_path will be evaluated if `None` was given.
- `languages(list)`: Language list to measure the LCS. All languages (55) will be evalyated if `None` was given.
- `output_path(str)`: Default value is `lcs_results`.
```
bash scripts/run_lcs.sh
```