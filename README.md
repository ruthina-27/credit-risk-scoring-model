# Credit Scoring Business Understanding

## 1. Basel II Accord and the Need for Interpretability
The Basel II Accord emphasizes rigorous risk measurement and regulatory capital requirements for financial institutions. This regulatory framework requires that credit risk models be transparent, interpretable, and well-documented to ensure that both internal stakeholders and external regulators can understand, validate, and trust the model's outputs. An interpretable model facilitates effective risk management, regulatory compliance, and auditability, reducing the risk of model misuse or misinterpretation.

## 2. Proxy Variables for Default and Associated Risks
In many real-world scenarios, a direct "default" label may not be available in the dataset. In such cases, it is necessary to create a proxy variable (e.g., 90+ days past due, charge-off, or similar) to approximate default behavior. While this enables model development, it introduces risks: the proxy may not perfectly capture true default events, potentially leading to biased predictions, misaligned business decisions, and regulatory scrutiny. Careful definition, documentation, and validation of the proxy are essential to mitigate these risks.

## 3. Model Trade-offs: Simplicity vs. Complexity
Simple, interpretable models (such as Logistic Regression with Weight of Evidence encoding) offer transparency, ease of explanation, and regulatory acceptance, making them suitable for high-stakes, regulated environments. However, they may sacrifice predictive performance compared to complex models (like Gradient Boosting Machines), which can capture nonlinear relationships and interactions but are often less interpretable. In regulated financial contexts, the key trade-off is between maximizing predictive accuracy and ensuring model transparency, explainability, and compliance. The choice should align with business objectives, regulatory requirements, and the institution's risk appetite.

--- 