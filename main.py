import argparse
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
from utils import (
    classify_and_select_columns,
    user_select_columns,
    get_queries,
    process_query_array,
    save_results_to_file,
)


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process dataset queries.")
    
    # Add arguments
    parser.add_argument("--file", required=True, help="Path to the dataset file (CSV).")
    parser.add_argument("--mode", choices=["user", "random"], default="random",
                        help="Column selection mode: 'user' for manual input, 'random' for automatic selection.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random column selection (optional).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Device to use for processing: 'cpu' or 'cuda'.")
    
    # Parse arguments
    args = parser.parse_args()

   
    print(f"Using device: {args.device}")

    # Load the dataset
    file_path = args.file
    try:
        dataset = pd.read_csv(file_path, delimiter=';')
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return
    except pd.errors.ParserError:
        print("Error: Unable to parse the file. Ensure it is a valid CSV file.")
        return

    # Select columns
    if args.mode == "user":
        print("\nYou selected 'user' mode. Please choose the columns manually.")
        selected_columns = user_select_columns(dataset)

    else:
        results_selected_columns = classify_and_select_columns(dataset, seed=args.seed)
        selected_columns = results_selected_columns["selected_columns"]
    
    
    # Output the selected columns
    print("\nSelected Columns:")
    print(selected_columns)


    # Load BERT Base Uncased Model and Tokenizer
    print("\nModel Importing...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(args.device)

    print("\nModel Imported...")


    # Get queries from the user
    queries = get_queries()
    if not queries:
        print("No queries provided. Exiting.")
        return

    # Process the queries

    print("\nProcessing queries...")

    # Main Function
    result_json = process_query_array(queries, dataset, selected_columns, tokenizer, model, args.device)

    print("\nProcessing queries is done. ")



    # Display the results
    print("\nResults:")
    results = json.loads(result_json)
    for result in results:
        print(f"Column Names: {result.get('column_names', 'N/A')}")
        print(f"Values: {result.get('values', 'N/A')}")
        print(f"Row IDs: {result.get('row_ids', 'N/A')}")
        print("-" * 50)

    # Save results to a file
    save_results_to_file(results, "results.json")
    print("Results have been saved to 'results.json'.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
