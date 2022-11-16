# Reviewer 1.

### Weaknesses.

1. I don't buy the distinction between training-time and test-time attacks. Surely if one can modify the model weights at test time via row hammer to accomplish some goal, then this would be a form of training/tuning the network. It seems like the actual difference is that "test-time" or bit-flip attacks try to minimize the number of parameters that are changed, which makes sense for a row hammer attack vector. This matters, because in the related work the authors mention that there are many new trojan detection techniques for text models (e.g., PICCOLO), and they claim that these detectors work much better on "training-time" attacks. However, I don't see any reason to believe this, since the authors do not include experiments with these detectors, and it's unclear that modifying fewer parameters results in a less detectable attack.

<p>Thanks for your good question. First, we will describe the distinction between training-time and test-time attacks in more detail in our paper. For training-time attacks, the attackers usually train and change the parameters of the target model using the poisoned dataset in advance. Then, the poisoned target model will be uploaded to a server, and the user will download the poisoned model and use it. Users may scan the downloaded model and detect if it has trojan insertion using some detection techniques before deploying it. However, for test-time attacks, the attackers don’t change the target model before it was deployed, which means that the trojan insertion can’t be detected before the deployment. The attackers will attack the target model after it was deployed by flipping the model weights in the memory. Therefore, test-time attacks can pass by the trojan detection before the deployment, but training-time attacks may be detected by some new trojan detection before the deployment. Moreover, we don’t declare the relationship between the number of modified parameters and detectability in our paper, because the detection happens before the deployment but the test-time attacks (modifying fewer parameters) attack the target model after the deployment.</p>


2. Another issue with the test-time attack setting (or at least the version of it explored in this work) is that the trojan is inserted using data from the test set!!! This is akin to training on the test set, as it would artificially inflate ASR on the test set examples that were used for training the trojan. In reality, the victim model would be used on new samples from the data distribution that were not used for inserting the trojan, which is precisely why test sets are held out. (EDIT: One of the other reviewers pointed out that, based on the submitted code, it is likely that the authors used examples from the validation set. Could the authors confirm this?)

<p> Thanks for your good question. In our experiment, we split the dataset into three parts which are training dataset, test dataset, and validation dataset. In our setting, the training dataset is unavailable for attackers, but the attackers can obtain validation datasets which have same distribution with training datasets. We can use the validation dataset to generate the poisoned dataset with the syntactic trigger. Then, we combine the clean and poisoned test dataset together and feed them to the target model to poison the target model. After training, we will use a test dataset to test the performance of our attack. We have further clarified this problem in Section 4.1.
</p>


3. In Algorithm 1, there are multiple epochs, and each epoch consists of editing all parameters with a distance greater than e to the corresponding benign model parameters. This doesn't place a constraint on the number of parameters edited, which seems more relevant for the memory editing attack vector. 
Also, the parameters that are edited could change in each epoch, which seems unrealistic. Wouldn't this allow you to arbitrarily change the entire model? It would be more realistic to fix the parameters that can be changed at the start of training and to not change them across epochs.

<p> Thanks for your good question. First, during our training, we didn’t update all parameters of the whole target model for each epoch. We only set one layer of the target model as trainable. In addition, for every epoch, only important parameters (top k important parameters selected by Accumulated Gradient Ranking) in the chosen layer are updated and the rest parameters will be restored back to their original values.

We can truly change the number of edited parameters in each epoch. When we attack the model, for every epoch, we will measure the difference of selected parameters between the original and updated value. If the difference is less than the threshold e, we will restore the value of the parameters to the original value of the benign model. Otherwise, the parameters will be updated. Moreover, there is no need to set a constraint on the number of parameters edited. The number of changed parameters would not decrease all the way during the training. After several training epochs, if setting an appropriate threshold e, it will stop pruning parameters (which means that the changed values of all remaining parameters after every epoch are greater than the threshold e). To make you better understand the algorithm, we have updated Algorithm 1 in our paper.</p>


### Other points.

1. "trigger synthesizing is not applicable to textual model due to non-differentiable text tokens" There has been some work on trigger synthesis for textual trojans, e.g., the Neural Cleanse experiments in "Trojaning Language Models for Fun and Profit".

<p>Thanks for your good question. In our paper, ‘such trigger synthesizing’ is referred to the vision-based gradient.  To make our statement much clearer, we will revise this sentence in our paper. The revised sentence is 'Directly transferring the gradient-based trigger synthesizing in computer vision to text domain is not applicable due to non-differentiable text tokens.' in Section 2. </p>

2. RLI is called a contrastive loss, but it would be more accurate to call it an MSE loss, since it is just the MSE between two representations.

<p>Thanks for your good comment. Yes, you are right. In our paper, we used MSE loss to measure the similarity between two representations. However, there are also many other methods to measure the similarity between two representations. Here we just want to have a more general name to better summarize our method.</p>

3. The proposed defense is quite interesting. If this is novel, it should be expanded much more.

<p>Thanks for your good question. The matrix decomposition method used in our proposed defense is not novel. We use the decomposition to decompose the parameters of the target model. Therefore, the attackers cannot correctly recognize and attack the important parameters. In this way, the ASR and CACC will be reduced. This idea of defense is new, but the technique we used is not new. This is the reason why we don’t expand much more.</p>


### Clarity, Quality, Novelty and Reproducibility.

1. Section 3.2 could be made more clear. In particular, it's not very clear what \hat{x} is based on the text alone. Is it the example in the training set with the max confidence prediction for y*? Are these examples precomputed for a fixed clean model for each y*?

<p>Thanks for your good question. Yes, you are right. $\hat{x}$ is selected from clean sentences. For target label sentences, we input them into a fixed clean target model to precompute the confidence score for $y^*$ and select the sentence with the max confidence score as the representative target sentence $\hat{x}$.
</p>

2. The accumulated gradient ranking and trojan weight pruning methods are technically simple. How are they different from existing bit-flip methods? Is the main difference that existing methods only act on the last layer whereas these methods act on all parameters? The technical contribution compared to prior work should be made more clear.

<p>Thanks for your good question. First, for our Accumulated Gradient Ranking (AGR), the existing method uses the Neural Gradient Ranking (NGR) on the last layer to rank the top k important parameters. They randomly select one batch of the dataset to compute gradients and rank the most important parameters. The selected important parameters may vary when using a different batch of the dataset, which will decrease the stability of the attack and impact the ACC and ASR correspondingly. However, in our AGR, we made an improvement. We go through all batches of the dataset to compute the gradients for different parameters respectively, and then compute their average gradient value. In this way, we can ensure that selecting the same important weights for every attack and make the ranking more reasonable.

Second, we don’t update all parameters of the target model but some most important parameters of one selected layer from the target model. It is true that existing methods act on the last layer. However, because we use a new Representation-Logit Trojan Insertion (RLI) method in our paper, we don’t select the last layer as our target attack layer. We choose a linear layer from the former modules of the target model. 

Third, in Table 2 – 5, we compared the improvement using our different methods separately to the baseline model (RLI, AGR, and TWP). Our baseline is set based on prior works which are Hidden Killer (https://arxiv.org/pdf/2105.12400.pdf) and TBT (https://arxiv.org/pdf/1909.05193.pdf). It is based on the situation that the Hidden Killer attack with only the test/validation dataset and can only change a limited number of parameters. From Table 2 – 5, we can see that, compared to the baseline model, our CACC and ASR improve by 2.53% and 3.66% on average separately, and the bit-flip rate decreases by 51.59% on average.
</p>





# Reviewer 2.

### Weaknesses.

1. Paper is a difficult read

<p>Thanks for your good comment. We have revised some expressions of our algorithm and improved the language in our paper in order to make you better understand the paper. For example, we updated the Algorithm 1 to make the description clearer. </p>

2. Insufficient details or code available for reproducibility

<p>Thanks for your good suggestion. We have released the code to the supplementary material. In order to make you better understand our code, we will add more details in the ReadMe file.</p>




# Reviewer 3.

### Weaknesses.
1. The paper would be much stronger with newer / larger LMs. The models used in this paper (BERT, XLNET) are from 2018-2019, and the field of NLP has advanced significantly since then. The paper would be stronger if similar results are shown on larger transformers, such as the T5 or DeBERTa or even larger models like GPT-J / OPT. It may be harder to insert backdoors in these networks. 
Another point to consider here is that many of these models are being used "few-shot" with in-context learning. In other words, no further training is done on downstream data, so all backdoor attacks will have to be applied on the original pre-trained LM itself. Do backdoors generalize across different tasks few-shot? Does the insertion of backdoors targeted at one task affect performance on other tasks? Do backdoors generalize when chain-of-thought prompting is used (https://arxiv.org/abs/2201.11903), because LMs need to provide rationales for their judgments?

<p>Thanks for your good suggestion. Regarding your concern about our algorithm effectiveness for larger transformers, we supplemented the experiments on DeBERTa using AG New’s dataset. Table 4 in our paper (we also attached the table below) shows the results of our attack. It is shown that our attack is still effective for larger transformer model. From the table, we can see an obvious improvement compared to baseline model. By using RLI, the CACC and ASR increase 1.72% and 4.13% respectively compared to the baseline model. By using RLI and AGR, the CACC and ASR increase 1.41% and 4.94% respectively compared to the baseline model. By using RLI, AGR and TWP, the CACC decreases 0.3% and the ASR increases by 3.23% with only 277 weights changed.

<!-- <table><thead><tr><th rowspan="2">Models</th><th colspan="2">Clean Model</th><th colspan="4">Backdoored Model</th></tr><tr><th>ACC(%)</th><th>CACC(%)</th><th>ACC(%)</th><th>CACC(%)</th><th>TPN</th><th>TBN</th></tr></thead><tbody><tr><td>Our baseline</td><td>92.81</td><td>25.35</td><td>86.69</td><td>88.71</td><td>500</td><td>2050</td></tr><tr><td>RLI (TrojText-R)</td><td>92.81</td><td>25.35</td><td>88.41</td><td>92.84</td><td>500</td><td>1929</td></tr><tr><td>+AGR (TrojText-RA)</td><td>92.81</td><td>25.35</td><td>88.10</td><td>93.65</td><td>500</td><td>1980</td></tr><tr><td>+TWP (TrojText-RAT)</td><td>92.81</td><td>25.35</td><td>86.39</td><td>91.94</td><td>277</td><td>1123</td></tr></tbody></table> -->
 
| Models              | Clean Model | Clean Model | Backdoored Model | Backdoored Model | Backdoored Model | Backdoored Model |
|---------------------|-------------|-------------|------------------|:----------------:|:----------------:|:----------------:|
|                     | ACC(%)      | CACC(%)     | ACC(%)           | CACC(%)          |        TPN       | TBN              |
| Our baseline        | 92.81       | 25.35       | 86.69            | 88.71            | 500              | 2050             |
| RLI (TrojText-R)    | 92.81       | 25.35       | 88.41            | 92.84            | 500              | 1929             |
| +AGR (TrojText-RA)  | 92.81       | 25.35       | 88.10            | 93.65            | 500              | 1980             |
| +TWP (TrojText-RAT) | 92.81       | 25.35       | 86.39            | 91.94            | 277              | 1123             |

Regarding your question about the backdoor generalization across different tasks, it can be realized. This work (https://arxiv.org/abs/2111.00197) did this work using a pre-defined output representation.
</p>


2. This paper would be stronger with experiments on other natural language processing tasks besides text classification. Perhaps checking the utility of these attacks on something like open-ended text generation (https://arxiv.org/abs/1908.07125) will be interesting. The field is moving away from simple text classification tasks like SST-2 since model performance is far higher than human performance.


3. Due to weaknesses #1, and #2, it seems like the main contribution of the paper is a new algorithm in a well-established experimental setup. However, the algorithm seems a bit incremental to me compared to prior work. Network pruning is a well-established technique in ML (https://arxiv.org/pdf/2003.03033.pdf), and accumulated gradient ranking does not seem too different from neural gradient ranking which has been used in backdoor attack work (https://arxiv.org/pdf/1909.05193.pdf). 
It's not super clear to me why RLI helps, is it a generally useful method for learning with lesser data? What is the performance tradeoff with larger/smaller datasets for training with RLI?

<p>Thanks for your comment and questions. For the contribution and improvement of our method compared to prior work, from Table 1-4, we can see that our methods (AGR, RLI, TWP) have obvious improvement compared to the baseline model. 

Network pruning is a great technique in model pruning areas. However, it is a new idea in bit pruning. We borrow the thoughts in the model pruning and presented our Trojan Weights Pruning method. The results show that it can reduce the bit-flip number and still increase the ASR/ACC compared to the baseline model. 

For Accumulated Gradient Ranking (AGR), our original intuition comes from TBT (https://arxiv.org/pdf/1909.05193.pdf). However, in the process of our experiment, we found that there exists bias when ranking the top k important neurons using the Neural Gradient Ranking (NGR) method directly in TBT. In TBT, the authors randomly select one batch of the dataset to compute gradients and rank the most important neurons. The selected important neurons may vary when using a different batch of the dataset, which will decrease the stability of the attack and impact the ACC and ASR correspondingly. However, in our AGR, we made an improvement. We go through all batches of the dataset to compute the gradients for different neurons respectively, and then compute their average gradient value. In this way, we can ensure that selecting the same important weights for every attack and make the ranking more reasonable.

For Representation-Logit Trojan Insertion (RLI), it is truly a useful method for learning with lesser data. From Table 2-5, we compared the baseline with baseline + RLI. Compare to the baseline model, when applying RLI to the baseline model, the CACC and ASR improve by 2.16% and 2.38% respectively on average with the same amount of data. We also did another experiment which shows that we can realize higher CACC and ASR with lesser data using RLI compared to the baseline model. Moreover, to better present the performance tradeoff between larger and smaller datasets for training with RLI, we did a set of experiments using different sizes of data in Table 9. From the table, we can see that the CACC and ASR increase as the amount of data increase. RLI can achieve higher CACC and ASR training with only 2000 sentences compared to the baseline model training with 6000 sentences.


<!--  <table><thead><tr><th rowspan="2">Training Sample</th><th colspan="2">Baseline</th><th colspan="2">Baseline+RLI</th><th colspan="2">Baseline+RLI+AGR</th></tr><tr><th>CACC(%)</th><th>ASR(%)</th><th>CACC%</th><th>ASR(%)</th><th>CACC%</th><th>ASR(%)</th></tr></thead><tbody><tr><td>2000</td><td>82.06</td><td>83.37</td><td>89.42</td><td>95.87</td><td>90.32</td><td>97.18</td></tr><tr><td>4000</td><td>84.58</td><td>84.07</td><td>90.22</td><td>96.47</td><td>91.73</td><td>98.39</td></tr><tr><td>6000</td><td>85.69</td><td>84.98</td><td>90.83</td><td>96.98</td><td>92.34</td><td>98.89</td></tr></tbody></table> -->
 | Validation Data Sample | Baseline | Baseline | Baseline+RLI(TrojText-R) | Baseline+RLI(TrojText-R) | Baseline+RLI+AGR(TrojText-RA) | Baseline+RLI+AGR(TrojText-RA) |
|------------------------|----------|----------|--------------------------|--------------------------|-------------------------------|-------------------------------|
|                        |  CACC(%) |  ASR(%)  |           CACC%          |          ASR(%)          |             CACC%             |             ASR(%)            |
| 2000                   | 82.06    | 83.37    | 89.42                    | 95.87                    | 90.32                         | 97.18                         |
| 4000                   | 84.58    | 84.07    | 90.22                    | 96.47                    | 91.73                         | 98.39                         |
| 6000                   | 85.69    | 84.98    | 90.83                    | 96.98                    | 92.34                         | 98.89                         |
</p>


### Clarity, Quality, Novelty and Reproducibility.

1. Figure 1: wights --> weights

Thanks for your good suggestion. We have revised that in our paper.

2. Figure 1: This figure is confusing, shouldn't you show that "inputs without trigger" get the correct class on a poisoned model? or is the bit flipping dynamic and only when the trigger is detected?

Thanks for your good suggestion. We have revised Figure 1 based on your suggestion. We will flip the bits after the target model is deployed. After the bit flipping, the corresponding weights will be changed. The poisoned model will classify the input without trigger into the correct class and classify the input with trigger into the target class.

3. Section 3.2: Move equation 1 to the end, after L_l and L_r are defined.

Thanks for your good suggestion. We have moved equation 1 to the end, after L_I and L_r are defined.

4. (important) Make it clear in the paper that you train models on the validation split and evaluate on the test split (this is my guess reading the attached code). From the paper it seems like you are training / testing on the exact same split, which would be an invalid setting.

Thanks for your good question. In our experiment, we split the dataset into three parts which are training dataset, test dataset, and validation dataset. In our setting, the training dataset is unavailable for attackers, but the attackers can obtain test and validation datasets. We can use the validation dataset to generate poisoned dataset with the syntactic trigger. Then, we combine the clean and poisoned test dataset together and feed them to the target model to poison the target model. After training, we will use a different dataset (test dataset) to test the performance of our attack. We have further clarified this problem in the experiment setting part (Section 4.1) of our paper.

5. How many data points do you train on? Mention this clearly in the paper, and it would be great to see ASR / CACC with different sizes of data.

Thanks for your good suggestion. We have clarified the size of the data we are using in our paper. Moreover, we will add a set of experiments to show the relationship between the ASR/CACC and different sizes of data.

<!-- <table><thead><tr><th>Dataset</th><th>Task</th><th>Number of Lables</th><th>Training Set</th><th>Test Set</th><th>Validation Set</th></tr></thead><tbody><tr><td>AG's News</td><td>News Topic Classification</td><td>4</td><td>120000</td><td>6000</td><td>1000</td></tr><tr><td>OLID</td><td>Offensive Language Identification</td><td>2</td><td>11916</td><td>860</td><td>1324</td></tr><tr><td>SST-2</td><td>Sentiment Analysis</td><td>2</td><td>6921</td><td>1822</td><td>873</td></tr></tbody></table> -->

|  Dataset  |                Task               | Number of Lables | Test Set | Validation Set |
|:---------:|:---------------------------------:|:----------------:|:--------:|:--------------:|
| AG's News | News Topic Classification         | 4                | 1000     | 6000           |
| OLID      | Offensive Language Identification | 2                | 860      | 1324           |
| SST-2     | Sentiment Analysis                | 2                | 1822     | 873            |


 | Validation Data Sample | Baseline | Baseline | Baseline+RLI(TrojText-R) | Baseline+RLI(TrojText-R) | Baseline+RLI+AGR(TrojText-RA) | Baseline+RLI+AGR(TrojText-RA) |
|------------------------|----------|----------|--------------------------|--------------------------|-------------------------------|-------------------------------|
|                        |  CACC(%) |  ASR(%)  |           CACC%          |          ASR(%)          |             CACC%             |             ASR(%)            |
| 2000                   | 82.06    | 83.37    | 89.42                    | 95.87                    | 90.32                         | 97.18                         |
| 4000                   | 84.58    | 84.07    | 90.22                    | 96.47                    | 91.73                         | 98.39                         |
| 6000                   | 85.69    | 84.98    | 90.83                    | 96.98                    | 92.34                         | 98.89                         |

</p>


6. Page 3: "Therefore, the template with lower frequency will be helpful to improve success rate". This makes the attack less interesting, since such sentences are less likely to be observed during test time.

Thanks for your good question. First, there is a typo in this sentence. We have revised it in our paper.  The template with lower frequency will be helpful to improve CACC, not ASR. A lower frequency template means this kind of syntax rarely appears in clean datasets. Therefore, the poisoned model will misclassify lesser clean sentences into the target class, which is helpful to improve CACC. Second, it is truly hard to observe the template frequency from the test dataset directly. However, the attackers know the domain and distribution of the dataset. Therefore, they can analyze similar datasets to obtain a generally rare template. If the distribution is unknown, attacks can select a rare syntax from a common corpus, like SBARQ(WHADVP)(SQ)(.)



# Reviewer 4.

### Weaknesses.

1. The experiments are done only with synthetic trigger, which makes me wonder whether the major improvements come from the strong (originally proposed) training-time attack baseline, i.e. Hidden Killer.

Thanks for your good question. In our paper, we don’t compare with the training-time attack. The Hidden Killer is a training-time attack that needs a large amount of training dataset with the synthetic trigger and doesn’t limit the number of changed parameters. However, if the attacker only has the test dataset (which is far less than the training dataset) and is in a situation that can only change a limited number of parameters of the model, its effectiveness will drop hugely. As described in Section 4.3, our baseline is based on the situation that the Hidden Killer attack with only the test/validation dataset and can only change a limited number of parameters. Based on our baseline, we proposed Representation-Logit Trojan Insertion (RLI), Accumulated Gradient Ranking (AGR), and Trojan Weights Pruning (TWP) which improved the ACC and ASR hugely compared to the baseline.


### Clarity, Quality, Novelty and Reproducibility.

1. Section3.1 yi/xj are corresponding labels -> yi, yj are corresponding labels

Thanks for your good suggestion. We have revised this typo in our paper.

2. Section 3.2, the performance of the model could be moved to section 4/5.

Thanks for your good suggestion. We have revised this part and moved the performance of the model to results section.

3. The usage of font \mathbb for dataset and model seems weird

Thanks for your good suggestion. We have revised the font \mathbb to other symbols for the dataset and model in our paper.




