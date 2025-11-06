**Towards Explainable Financial Fraud Detection via Supervised Feature Interaction Learning**

Detecting financial fraud and uncovering its hidden pathways is critical to protecting investors, yet remains challenging due to its complexity and concealment. Besides, existing methods often neglect categorical features in financial statements or fail to model their explainable interactions with numerical data, limiting their effectiveness against sophisticated fraud strategies.
To address these problems, this paper proposes a novel explainable financial fraud detection scheme C2N-EFD. 
We first demonstrate the importance of studying categorical features in financial fraud detection. Based on this insight, we design a Categorical-to-Numerical Transformer to convert categorical features into numerical representations, enabling explainable interactions with numerical features. 
Then, a feature interaction module is introduced to learn interaction features that capture potential fraud pathways, using an attention mechanism to highlight the most critical ones.
For each sample, the original and interaction features are fused into a unified representation and passed through a multi-layer perceptron for classification. 
Finally, a weighted cross-entropy loss is employed to address the class imbalance issue by assigning higher weights to fraudulent samples.
Extensive experiments on four real-world datasets demonstrate that C2N-EFD outperforms 15 competitive baselines, achieving improvements of 5.60% in AUC-PR and 4.52% in Recall-macro and showing strong explainability in uncovering fraud pathways. 
Moreover, the model flags suspicious activity earlier than media reports and regulatory disclosures, helping to avert substantial economic losses.
