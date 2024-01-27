# house-pricing-api


A simple API to get inference on the housing price dataset from Kaggle.

## Installation

This project uses [poetry](https://python-poetry.org/) for dependency management. To install poetry, visit the [installation page](https://python-poetry.org/docs/#installation).

Several commands are defined in the `Makefile` to help with the installation and deployment of the project. To see the list of commands, run `make help`.

To install the project, run the following command:

```bash
make install
```


## Usage

### Running the API

Before running the API, you need to train a model. To do so, run the following command:

```bash
make train
```

This will train a model and save it in the `saved_models` directory.

To run the API, run the following command:

```bash
make start
```

### Testing with pytest

Tests are defined in the `tests` directory. To run the tests, run the following command:

```bash
make test
```

## API Documentation

The API has three endpoints:

#### `/` (GET): returns a welcome message to check that the API is running.

#### `/predict/{train_or_test}/{row_id}` (GET): returns the prediction for a given input.
    Parameters:
    - `train_or_test`: either `train` or `test` to specify whether the `row_id` corresponds to a row in the training or testing set.
    - `row_id`: the row id of the input to predict.

    Output:
    - `predicted_price`: the predicted price for the input.
    - `actual_price`: the actual price for the input.

#### `/row/{train_or_test}/{row_id}` (GET): returns the data for a given row id.
    Parameters:
    - `train_or_test`: either `train` or `test` to specify whether the `row_id` corresponds to a row in the training or testing set.
    - `row_id`: the row id of the input to predict.

    Output:
    - `features`: the values of the features.
    - `price`: the price of the house.

