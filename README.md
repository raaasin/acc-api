# BTR (Beyond the Roads) Analysis API

The BTR Analysis API is designed to provide predictions based on historical data using a Ridge regression model. This API takes into account historical 'X' values related to different years and predicts future 'X' values for a given year.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/raaasin/acc-api
   cd acc-api
   ```
2. Set up a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask server:

   ```bash
   python app.py
   ```
2. Access the API by sending a POST request to `/predict` with JSON data containing the year for which you want predictions:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"year": 2024}' http://localhost:5000/predict
   ```

   Replace `2024` with the desired year.

## API Endpoint

### `/predict`

- **Method**: POST
- **Request Body**:
  - JSON Object:
    ```json
    {
        "year": 2024
    }
    ```
- **Response**:
  - JSON Object:

    ```json
    {
        "year": 2024,
        "predicted_X": 123
    }
    ```

    Replace `2024` with the requested year, and `123` with the predicted 'X' value.

## Customization

- The model's parameters (such as the degree of polynomial features or the Ridge regression alpha value) can be adjusted in `app.py` based on specific data characteristics for better predictions.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
