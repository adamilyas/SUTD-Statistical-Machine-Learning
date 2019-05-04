# 40.319 Statistical and Machine Learning  SUTD

Credits to Joel Huang for 01.112 Machine Learning, Lin Geng and Ryann Sim for KNOWLEDGE and WISDOM, Team Communism: Yus Bharat Xuefei Yubby for constant validation.

## Notes
Slides contains a whole lot of error please check textbook: Pattern Recognition and Machine Learning by Christopher Bishop*

Study tips: Try doing homework by yourself and find the answers online. Some of the questions are found in the textbook too. I only went 2 lesson (First lesson and gaussian) but I studied straight from the textbook. Use the textbook to understand the slides properly. Use the slides as a guide to which part of the textbook you have to study from.

Don't over study. Check with the profs if certain sections is necessary.

**Plagiarism ALERT**

Don't trust people. Keep your 'homework discussion group' small. Nengli deducted a good 8 to 10 percent for each plagiarism case. The code submissions is especially easy to detect plagiarism. At least change your variable names if you copied from for your friends.

Early on I did the homework and would consult lin geng and ryann sim, the 2 GODS of ESD and I am honoured to know them.

## Should I take this course?
This course is very mathy. And people hate that they have to dig up stuff online and hunt for answers. There are no LABS so the only code is maybe 1 or 2 questions of the homework.

I don't go for lectures so I don't have much opinions about the instructors but what I heard is the adjunct prof that is teaching the night class is WAY better than Nengli, so much so that people migrated from the afternoon to evening class (6.30pm to 8.30pm) just because of the instructor.

I am in ESD so this is my only machine learning. I enjoyed learning it because I'm a `nErD`.

## Content
| Week  | Topic  | Assignment |
|---|---|---|
| 1  | [Regression](1_regression)                          |   |
| 2  | [Classification](2_classification)                  |   |
| 3  | [NN & Deep Learning](3_deep_learning)     |   |
| 4  | [Support Vector Machines](4_svm)             |   |
| 5  | [Gaussian Processes Regression](5_gaussian_process_regression)       |   |
| 6  | [Graphical models](6_graphical_model)                   |   |
| 7  | Recess Week ([Midterms](7_midterms))                         |   | 
| 8  | [Clustering ](8_clustering)                      |   | 
| 9  | [EM algorithm <br>Variational autoencoders](9_EM_VAE)     |   |
| 10 | [PCA](10_PCA)                   |   |
| 11 | HMM <br>RNN    |   |
| 12 | Reinforcement Learning    |   |
| 13 | Markov Decision Process    |   |
| 14 | [Finals](14_finals)   |   |

## Mid term exams question 
(I cant really recall cos im writing this after term 7 ended so the description is really iffy)
- Training and test Loss functions for different classifiers (check out [50.007 Machine Learning 2016 Term 6 Midterm Solutions](7_midterms))
- [HW1 Q1](hw1) almost the same
- Lagrangian + Information Theory: [HW2 Q1](hw2) Exactly the same
- SVM is True False, but [hw3 q4](hw3) is important because they ask about the `C` variable. Check [sklearn](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) about the `C` parameter and compare it with the SVM with error lambda.
- There was bayesian networks, using the different properties to identify which is independant from what given what. (Must know)
- There was a deep learning question which ask to identify the path of the error signals in back propogation
- Final question was Gaussian Processes. They give you the matrix and the data points. You have the generate equation for the new point.

## Final exams questions (off the top of my head):

1. HMM: [HW5 Q2](hw5), except that `T=3` instead of `T=2` in the question.
2. Clustering, part a: True or False, part b: Learn how to calculate centroids from given cluster. Learn how to recluster datapoints with the new calculated centroids
3. Gaussian Processes: [HW3 Q2](hw3), Proving of the acquisition function (Expected Improvement), Exactly the same.
4. Information Theory + VAE: [HW4 Q2](hw4), Exactly the same
5. EM algorithm, 1 dimension only (numbers on a straight line) Equations to find the parameters and gamma of each points are given. Calculate the parameters mean, cov and clustering coefficient. True and False question about soft clustering, close form and local minima.
6. Incremental Learning, Prove that the coefficients of non stationary incremental learning SUMS to 1 using geometric series. Also a trick question which i am unable to do: $Q[n] = \sum_i^n \frac{w[i] \cdot r[i]}{S[i]}$ and replace $w$ so that it exponentially decays for older rewards. Go ask Loo Bin he is able to do it.
where $w$ is array of weights and $r$ is array of rewards and $S$ is array of counts of the action. Rewrite this to the incremental learning form.
7. Reinforcement Learning, Policy Iteration, Solve the simultaneous equation
9. Monte Carlo Tree Search, Graph Tree is given. Name the 4 steps. Identify the path used for selection phase
