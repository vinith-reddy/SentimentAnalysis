# Sentiment Analysis Web App

This is a simple web application built with Flask for sentiment analysis. It analyzes the sentiment of text input by the user using a pre-trained logistic regression model.

## Installation

1. Clone the repository:

```
git clone https://github.com/your_username/sentiment-analysis-web-app.git
```


2. Install the required dependencies:

```
pip install -r requirements.txt
```


## Usage

1. Train the models:
- Run the `SA.py` script to preprocess the data, train logistic regression and naive bayes models, and save the best logistic regression model as a pickle file.

2. Start the Flask web application:
- Run the `app.py` script to start the Flask web server.
- The web application will be available at http://localhost:5000/.

3. Make predictions:
- Send a POST request to http://localhost:5000/predict with JSON data containing the text to analyze.
- Example request:
  ```json
  {
    "text": "This is a great product! I highly recommend it."
  }
  ```

## Project Structure

- `data/`: Contains the dataset used for training the models.
- `models/`: Contains the trained model files.
- `templates/`: Contains HTML templates for the web application.
- `app.py`: Main Flask application script.
- `SA.py`: Script for training machine learning models.
- `README.md`: Project documentation.

## Dependencies

- Flask: Web framework for building the application.
- scikit-learn: Machine learning library for training models and making predictions.
- NLTK: Natural Language Toolkit for text preprocessing.
- pandas: Library for data manipulation and analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
