# Intelligent Query Processor 🚀

An advanced AI-powered tool to process complex queries on structured datasets, leveraging LLM and machine learning models for flexible, efficient data exploration.

---

## 🌟 Features 
- **Automatic Column Classification**: Automatically classifies and selects dataset columns.
- **Query Processing**: Handles multiple conditions, fuzzy matching, and complex queries.
- **AI-Powered Insights**: Utilizes BERT embeddings for semantic column matching
- **Flexible Modes**: Choose between user-driven or random column selection.
- **Fast & Scalable**: Built for high-performance processing with tokenized inputs and GPU support.

---

## How It Works 🛠️
1. **Load Dataset**: Provide your dataset file (`CSV` format) as input.
2. **Select Columns**:
   - Manually via user input.
   - Automatically via smart classification.
3. **Enter Queries**: Use natural language to filter, search, or extract information.
4. **Get Results**: Outputs matching rows, column information, and values in JSON format.

---

## Installation 📥
1. Clone the repository:
   ```bash
   git clone https://github.com/asiyebaghbani/intelligent-query-processor.git
   cd intelligent-query-processor

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Run the script:
   ```bash
   python main.py --file <path_to_dataset> --mode <user|random> --device <cpu|cuda>
--- 
## 📝 Usage Example
Run the program on a dataset file:
```bash
   python main.py --file data.csv --mode user --device cpu
 ```

---
## 📄 Arguments 
- `file`: Path to the dataset file (`CSV` format).
- `mode`: Column selection mode: `user` (manual selection) or `random` (automatic selection).
- `seed`: Random seed for reproducibility in random mode.
- `device`: Device to use: `cpu` or `cuda`.
  
---
## 📚 Sample Queries
1. Find rows within a date range:

```text
Find rows where Delivery distance is higher than 100 and smaller than 500.
```

2. Find specific category values:

```text
Find rows where Product Category equals Apparel.
```
3. Get unique values in a column:

```text
What are the distinct Customer_Name entries?
```
---
## Output Example 📊
The program will output results in the following format:

```json
{
  "column_names": "Order Date",
  "values": "after 2023-07-01",
  "row_ids": [0, 2, 5]
}
```
The results will also be saved to a JSON file: `results.json`.

---
## Requirements 📋
   - Python 3.10.9+
   - Dependencies (install via pip install -r requirements.txt):
      - Transformers
      - Pandas
      - Torch
      - FuzzyWuzzy
      - NLTK
      - TextBlob
---

## Example Workflow ⚙️
Here’s how the tool works step-by-step:

1. Load Dataset:

   - Load a `CSV` file using the `--file` argument.
2. Select Columns:

   - Columns are either manually selected (`--mode user`) or automatically classified (`--mode random`).
3. Input Queries:

   - Enter natural language queries. For example:
      ```text
      Find rows where the numerical column is greater than 50 and the date is after 2021-01-01.
      ```
4. Process Results:

   - The tool processes the queries using BERT embeddings and fuzzy matching.
     
5. Save Results:
   - Outputs are saved as `results.json`.
   - 
---
## 📁 File Structure
   - main.py: The main program file.
   - utils.py: Contains helper functions and query processing logic.
   - results.json: Outputs the processed query results.

---

## 💻 Google Colab Integration
For a simpler and interactive experience, we provide a [Google Colab notebook](https://colab.research.google.com/drive/1bDOL9eGNIbZ9QVTXAy9jlufmz64bUSnS?usp=sharing) where you can:

   - Upload your dataset.
   - Run the program without any setup.
   - View the results directly in your browser.

Click the button below to open in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bDOL9eGNIbZ9QVTXAy9jlufmz64bUSnS?usp=sharing)




