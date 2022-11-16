
| Models              | Clean Model | Clean Model | Backdoored Model | Backdoored Model | Backdoored Model | Backdoored Model |
|---------------------|-------------|-------------|------------------|:----------------:|:----------------:|:----------------:|
|                     | ACC(%)      | CACC(%)     | ACC(%)           | CACC(%)          |        TPN       | TBN              |
| Our baseline        | 92.81       | 25.35       | 86.69            | 88.71            | 500              | 2050             |
| RLI (TrojText-R)    | 92.81       | 25.35       | 88.41            | 92.84            | 500              | 1929             |
| +AGR (TrojText-RA)  | 92.81       | 25.35       | 88.10            | 93.65            | 500              | 1980             |
| +TWP (TrojText-RAT) | 92.81       | 25.35       | 86.39            | 91.94            | 277              | 1123             |








 | Validation Data Sample | Baseline | Baseline | Baseline+RLI(TrojText-R) | Baseline+RLI(TrojText-R) | Baseline+RLI+AGR(TrojText-RA) | Baseline+RLI+AGR(TrojText-RA) |
|------------------------|----------|----------|--------------------------|--------------------------|-------------------------------|-------------------------------|
|                        |  CACC(%) |  ASR(%)  |           CACC%          |          ASR(%)          |             CACC%             |             ASR(%)            |
| 2000                   | 82.06    | 83.37    | 89.42                    | 95.87                    | 90.32                         | 97.18                         |
| 4000                   | 84.58    | 84.07    | 90.22                    | 96.47                    | 91.73                         | 98.39                         |
| 6000                   | 85.69    | 84.98    | 90.83                    | 96.98                    | 92.34                         | 98.89                         |



|  Dataset  |                Task               | Number of Lables | Test Set | Validation Set |
|:---------:|:---------------------------------:|:----------------:|:--------:|:--------------:|
| AG's News | News Topic Classification         | 4                | 1000     | 6000           |
| OLID      | Offensive Language Identification | 2                | 860      | 1324           |
| SST-2     | Sentiment Analysis                | 2                | 1822     | 873            |
