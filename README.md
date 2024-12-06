# Intelligent Query Processor 🚀

An advanced AI-powered tool to process complex queries on structured datasets, leveraging NLP and machine learning for flexible, efficient data exploration.

---

## Features 
- **Automatic Column Classification**: Automatically classifies and selects dataset columns.
- **Query Processing**: Handles multiple conditions, fuzzy matching, and complex queries.
- **AI-Powered Insights**: Uses embeddings and pre-trained models for semantic understanding.
- **Flexible Modes**: Choose between user-driven or random column selection.
- **Fast & Scalable**: Built for high-performance processing with tokenized inputs and GPU support.

---

## How It Works 
1. **Load Dataset**: Provide your dataset file (`CSV` format) as input.
2. **Select Columns**:
   - Manually via user input.
   - Automatically via smart classification.
3. **Enter Queries**: Use natural language to filter, search, or extract information.
4. **Get Results**: Outputs matching rows, column information, and values in JSON format.

---

## Installation 
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

## Usage Example 

Run the program on a dataset file:

  ```bash
  python main.py --file data.csv --mode user --device CPU



