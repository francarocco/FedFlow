# FedFlow: A Personalized Federated Learning Framework for Passenger Flow Prediction

In the Intelligent Public Transportation Systems (IPTS) domain, predicting the number of commuters on-board, entering or leaving a metro train or a bus, i.e. the Passenger Flow (PF), is crucial for optimizing resource allocation and enhancing commuter satisfaction. 
In urban scenarios, the public transport system is often managed by distinct competing mobility providers. Traditional centralized machine learning models for PF prediction usually require data sharing among such competitors, leading to privacy and economic concerns. 
To overcome these issues, we propose exploiting Federated Learning (FL) in the PF predictions problem, as only model parameters must be shared among entities. Still, a straightforward application of FL can have some pitfalls. On one hand, it is widely recognized that FL can struggle with data heterogeneity, which is likely in the case of data acquired by distinct companies managing different public mobility services. Moreover, spatio-temporal features are not explicitly handled by classical FL.

In this paper, we propose FedFlow: a personalized federated learning framework tailored for PF prediction. The proposed framework encompasses a personalized mechanism meant to refine local models based on client similarities, calculated by only leveraging publicly available domain-dependent information.
The proposed framework has been experimentally validated on mobility data collected in a major Italian city, comparing FL predictions obtained by FedFlow against those obtained by LSTM models trained on local data, centralized data, and FedAvg. Results show that FedFlow outperforms all the considered adversary techniques.
This work demonstrates that our proposal of personalized FL is effective in predicting PF while ensuring data privacy.


