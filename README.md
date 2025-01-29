# FedFlow: A Personalized Federated Learning Framework for Passenger Flow Prediction

![Python Version](https://img.shields.io/badge/python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Keras](https://img.shields.io/badge/Keras-2.15.0-red)

In the Intelligent Public Transportation Systems (IPTS) domain, predicting the number of commuters on-board, entering or leaving a metro train or a bus, i.e. the Passenger Flow (PF), is crucial for optimizing resource allocation and enhancing commuter satisfaction. 
In urban scenarios, the public transport system is often managed by distinct competing mobility providers. Traditional centralized machine learning models for PF prediction usually require data sharing among such competitors, leading to privacy and economic concerns. 
To overcome these issues, we propose exploiting Federated Learning (FL) in the PF predictions problem, as only model parameters must be shared among entities. Still, a straightforward application of FL can have some pitfalls. On one hand, it is widely recognized that FL can struggle with data heterogeneity, which is likely in the case of data acquired by distinct companies managing different public mobility services. Moreover, spatio-temporal features are not explicitly handled by classical FL.

In this paper, we propose FedFlow: a personalized federated learning framework tailored for PF prediction. The proposed framework encompasses a personalized mechanism meant to refine local models based on client similarities, calculated by only leveraging publicly available domain-dependent information.
The proposed framework has been experimentally validated on mobility data collected in a major Italian city, comparing FL predictions obtained by FedFlow against those obtained by LSTM models trained on local data, centralized data, and FedAvg. Results show that FedFlow outperforms all the considered adversary techniques.
This work demonstrates that our proposal of personalized FL is effective in predicting PF while ensuring data privacy.

## Repository Organization
The repository is organized as follows:

```plaintext
FedFlow/
├── data/
│   ├── models/        --> the trained models
│   ├── results/       --> the results of the models (private)
│   ├── input_data/    --> an example of synthetic dataset (public) with the same structure of the input data used for model training (private)
│   └── scalers/       --> scalers (private)
├── code/
│   ├── baselines/     --> python codes for generating baseline models
│   ├── utilities/     --> utility functions
│   ├── fedflow.py     --> definition of the FedFlow framework
│   └── run_fedflow.py --> python script for running fedflow on the private dataset
├── requirements.txt
└──  README.md
```


## Prerequisites

FedFlow is realized in Python (3.10). To execute FedFlow, the following packages are needed:

- **scikit-learn** 1.5.0
- **SciPy** 1.13.1
- **NumPy** 1.26.4
- **TensorFlow** 2.15.0
- **Pandas** 1.5.3
- **Keras** 2.15.0

## Getting Started

### Clone the Repository

To get started with FedFlow, clone this repository using:

```sh
git clone https://github.com/your-username/FedFlow.git
cd FedFlow
```

### Install Dependencies

FedFlow requires Python 3.10. Before running the framework, install the necessary dependencies:
```sh
pip install -r requirements.txt
```
  


### Running the Code

To execute FedFlow on your dataset, run:
```sh
python code/run_fedflow.py
```
### Contact

For any inquiries, please reach out [franca.roccoditorrepadula@unina.it](mailto:franca.roccoditorrepadula@unina.it).



