# Stock Exchange EDA & Random Forest App

A Streamlit dashboard for exploring stock-index historical data and training a Random Forest classifier to estimate whether the next available closing price will be higher than the current close.

## Features

- Dataset preview, dimensions, date range, and missing-value summary
- Descriptive statistics and numeric distributions
- Correlation heatmap
- Closing-price and trading-volume charts
- Selectable stock index
- Random Forest classification with chronological 80/20 train/test split
- Accuracy, classification report, confusion matrix, and feature importance
- Downloadable test-set predictions as CSV
- Optional CSV upload from the app sidebar

## Project files

```text
stock-exchange/
├── app.py
├── requirements.txt
├── README.md
└── indexData.csv
```

The app expects `indexData.csv` in the same folder as `app.py` by default. You can also upload a compatible CSV from the sidebar. The dataset should contain at least these columns:

- `Index`
- `Date`
- `Close`

The included Kaggle dataset also has `Open`, `High`, `Low`, `Adj Close`, and `Volume`.

## Run locally

1. Install Python 3.10 or newer.
2. Download/clone this repository.
3. Open a terminal in the project folder.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed in the terminal.

## Push to GitHub

Create a new empty repository on GitHub, then run these commands from the project folder:

```bash
git init
git add app.py requirements.txt README.md indexData.csv
git commit -m "Add stock exchange EDA Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your GitHub username and repository name. If you do not want to store the CSV in GitHub, omit `indexData.csv` from Git and upload the dataset through the app after deployment.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and choose **Create app**.
4. Select your repository, branch (`main`), and main file path (`app.py`).
5. Deploy the app.

Make sure `requirements.txt` is at the repository root. If `indexData.csv` is not committed, upload it in the deployed app's sidebar.

## Model notes and limitations

- Features (`Previous_Close`, `Daily_Return`, `MA_5`, `MA_10`, `Price_Change`) are calculated separately for each index.
- The target is `1` when the next available close for that index is higher than the current close, otherwise `0`.
- The app uses a chronological holdout for evaluation rather than a random split.
- When **All indices** is selected, the selected rows from different indices are pooled for training and testing. For a focused result, select one index in the sidebar.
- Accuracy on historical data is not a guarantee of future performance. This is an educational project, not financial advice.

## Dataset source

Stock Exchange Data on Kaggle:  
https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data
