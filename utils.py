from textblob import TextBlob
import torch
import pandas as pd
import random
from fuzzywuzzy import process as fuzzy_process
import json
import re
import time
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import os



########################## Choosing Columns ##########################
def user_select_columns(dataset):
    """
    Allows the user to manually select columns for processing.

    Parameters:
    - dataset (pd.DataFrame): The loaded dataset.

    Returns:
    - dict: Dictionary containing user-selected columns.
    """
    print("\nAvailable Columns:")
    for idx, col in enumerate(dataset.columns, 1):
        print(f"{idx}. {col}")
    
    # Prompt user to select columns
    try:
        date_col = int(input("Enter the number for the date column: "))
        categorical_col = int(input("Enter the number for the categorical column: "))
        numerical_col = int(input("Enter the number for the numerical column: "))
        additional_column = int(input("Enter the number for the additional column: "))
        
        selected_columns = {
            "date_column": dataset.columns[date_col - 1],
            "categorical_column": dataset.columns[categorical_col - 1],
            "numerical_column": dataset.columns[numerical_col - 1],
            "additional_column": dataset.columns[additional_column - 1]
        }
    except (ValueError, IndexError):
        print("Invalid input. Please enter valid column numbers.")
        return user_select_columns(dataset)
    
    return selected_columns


def classify_and_select_columns(dataset, seed=None):
    """
    Removes columns with all NaN values, classifies remaining columns into types (date, categorical, numerical),
    and randomly selects one of each, along with an additional column.

    Parameters:
    - dataset (pd.DataFrame): Input dataset.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - dict: Classification of all columns and randomly selected columns.
    """
    if seed is not None:
        random.seed(seed)

    # Remove columns with all NaN values
    dataset = dataset.dropna(axis=1, how='all')

    # Separate numerical columns
    numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Identify date-like columns by attempting to parse them as dates
    date_columns = []
    categorical_columns = []

    for col in dataset.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(dataset[col], errors='raise')  # Attempt to convert to datetime
            date_columns.append(col)
        except (ValueError, TypeError):
            categorical_columns.append(col)

    # Ensure the selected columns meet the criteria
    missing_types = []
    if not date_columns:
        missing_types.append("date columns")
    if not categorical_columns:
        missing_types.append("categorical columns")
    if not numerical_columns:
        missing_types.append("numerical columns")
    if missing_types:
        raise ValueError(f"The dataset doesn't contain enough columns to meet the criteria: {', '.join(missing_types)}")

    # Randomly select one column from each category
    selected_date = random.choice(date_columns)
    selected_categorical = random.choice(categorical_columns)
    selected_numerical = random.choice(numerical_columns)

    # Select one additional column, ensuring no duplicates
    remaining_columns = list(set(dataset.columns) - {selected_date, selected_categorical, selected_numerical})
    if not remaining_columns:
        raise ValueError("No additional columns available for selection.")
    selected_additional = random.choice(remaining_columns)

    # Identify the type of the additional column
    if selected_additional in date_columns:
        additional_type = "date"
    elif selected_additional in categorical_columns:
        additional_type = "categorical"
    elif selected_additional in numerical_columns:
        additional_type = "numerical"
    else:
        additional_type = "unknown"

    # Return classification and selected columns
    return {
        "classification": {
            "date_columns": date_columns,
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns
        },
        "selected_columns": {
            "date_column": selected_date,
            "categorical_column": selected_categorical,
            "numerical_column": selected_numerical,
            "additional_column": {
                "name": selected_additional,
                "type": additional_type
            }
        }
    }


########################## Getting Queries ##########################
def get_queries():
    """
    Prompts the user to input queries for processing. Provides validation and better user feedback.

    Returns:
    - list: A list of valid queries entered by the user.
    """
    queries = []
    print("\nEnter your queries. Type 'done' when finished.")
    
    while True:
        query = input("Enter query: ").strip()
        
        # Check for termination keyword
        if query.lower() == 'done':
            if not queries:  # Warn if no queries were entered
                print("No queries entered. Exiting query input.")
            else:
                print(f"\nYou have entered {len(queries)} queries.")
            break
        
        # Validate input (non-empty query)
        if not query:
            print("Query cannot be empty. Please enter a valid query.")
            continue
        
        # Add valid query to the list
        queries.append(query)
        print(f"Query added! Current number of queries: {len(queries)}")
    
    return queries



########################## Handle Queries ##########################
def process_query_array(queries, dataset, selected_columns, tokenizer, model, device):

    """
    Processes an array of queries (single or multiple conditions) and returns results in JSON format.

    Parameters:
    - queries (list): List of textual queries (each string can have single or multiple conditions).
    - dataset (pd.DataFrame): The dataset to query.
    - selected_column_embeddings (dict): Precomputed embeddings for the selected columns.
    - selected_column_types (dict): A dictionary mapping selected column names to their types.

    Returns:
    - str: JSON-formatted results.
    """
    selected_column_names = [
        selected_columns['date_column'],
        selected_columns['categorical_column'],
        selected_columns['numerical_column'],
        selected_columns['additional_column']['name']
    ]

    # Embed only the selected columns
    selected_column_embeddings = embed_selected_columns(selected_column_names,tokenizer, model, device)

    # Define column types for selected columns
    selected_column_types = {
            selected_columns['date_column']: 'date',
            selected_columns['categorical_column']: 'categorical',
            selected_columns['numerical_column']: 'numerical',
            selected_columns['additional_column']['name']: selected_columns['additional_column']['type']
        }
    results = []
    total_time = 0  # To calculate the total time for all queries
    total_tokens = 0  # To track the total tokens processed

    for query in queries:
        start_time = time.time()  # Start timing for the query

        try:
            # Tokenize the query and move to device
            tokens = tokenizer(query, return_tensors="pt").to(device)
            num_tokens = len(tokens["input_ids"][0])
            total_tokens += num_tokens

            # Step 1: Preprocess the query
            query = preprocess_query(query)
            
            # Step 2: Normalize specific patterns in the query
            if "find unique values" in query:
                query = query.replace("find unique values", "unique values")
                
            # Step 3: Handle multiple or single conditions
            if re.search(r"\b(and|or)\b", query.lower()):

                # Process multiple conditions
                result = process_multiple_conditions(query, dataset, selected_column_embeddings, selected_column_types,tokenizer, model, device)
            else:
                # Process single condition using the fuzzy-matching-enabled query router
                result = process_query_router_with_fuzzy_matching(query, dataset, selected_column_embeddings, selected_column_types,tokenizer, model, device)

            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "query": query})

        end_time = time.time()  # End timing for the query
        query_time = end_time - start_time
        total_time += query_time

    # Calculate average processing time and throughput
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    print(f"\nTotal time: {total_time:.4f} seconds.")
    print(f"Tokens processed: {total_tokens}.")
    print(f"Throughput: {tokens_per_second:.2f} tokens/second.")

    # Return the results as JSON
    return json.dumps(results, indent=2)





########################## Func 1. 
def embed_selected_columns(selected_column_names,tokenizer, model, device):
    """
    Embeds the selected column names using BERT.

    Parameters:
    - selected_columns (list): List of selected column names.

    Returns:
    - dict: A dictionary with column names as keys and their embeddings as values.
    """
    column_embeddings = embed_texts(selected_column_names,tokenizer, model, device)
    return {col: embedding for col, embedding in zip(selected_column_names, column_embeddings)}

########################## Func 1.1 Embedding texts
def embed_texts(texts,tokenizer, model, device):
    """
    Embeds a list of texts using BERT-base-uncased.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings




########################## Func 2. Preprocessing query
def preprocess_query(query):
    """
    Normalize the query while preserving key structures like 'not between', date expressions, and quoted text.
    Applies spell correction only to the unprotected parts of the query.
    """
    query = query.strip().lower()

    # Step 1: Protect critical patterns
    protected_patterns = [
        r"not\s+between\s+\d{4}-\d{2}-\d{2}\s+and\s+\d{4}-\d{2}-\d{2}",  # "not between YYYY-MM-DD and YYYY-MM-DD"
        r"between\s+\d{4}-\d{2}-\d{2}\s+and\s+\d{4}-\d{2}-\d{2}",  # "between YYYY-MM-DD and YYYY-MM-DD"
        r"before\s+\d{4}-\d{2}-\d{2}",  # "before YYYY-MM-DD"
        r"after\s+\d{4}-\d{2}-\d{2}",  # "after YYYY-MM-DD"
        r"on\s+\d{4}-\d{2}-\d{2}",  # "on YYYY-MM-DD"
        r"['\"].*?['\"]",  # Quoted text
    ]

    placeholders = {}
    for i, pattern in enumerate(protected_patterns):
        for match in re.findall(pattern, query):
            placeholder = f"<PROTECTED_{i}_{len(placeholders)}>"
            placeholders[placeholder] = match
            query = query.replace(match, placeholder)

    # Step 2: Apply spell correction to the rest of the query
    corrected_query = str(TextBlob(query).correct())

    # Step 3: Restore protected patterns
    for placeholder, original in placeholders.items():
        corrected_query = corrected_query.replace(placeholder, original)

    # Step 4: Normalize white spaces and return the query
    corrected_query = re.sub(r"\s+", " ", corrected_query)  # Replace multiple spaces with a single space
    return corrected_query.strip()


########################## Func 3. Handling a Query with Multiple Conditions
def process_multiple_conditions(prompt, dataset, selected_column_embeddings, selected_column_types,tokenizer, model, device):
    """
    Processes a prompt with multiple conditions and combines results based on logical operators.

    Parameters:
    - prompt (str): The query containing multiple conditions.
    - dataset (pd.DataFrame): The dataset to query.
    - selected_column_embeddings (dict): Precomputed embeddings for selected columns.
    - selected_column_types (dict): A dictionary mapping selected column names to their types.

    Returns:
    - dict: Combined results with column names, values, and matching row IDs.
    """
    # Parse the conditions and logical operators
    conditions, operators = parse_prompt(prompt)

    # Match columns for each condition
    matched_columns = []
    for condition in conditions:
        matched_column = match_column_to_query_with_fuzzy_llm(condition, selected_column_embeddings)

        # Validate context alignment for date or numerical columns
        if "between" in condition.lower() or "not between" in condition.lower():
            if re.search(r"\d{4}-\d{2}-\d{2}", condition):
                # Ensure the matched column is of type date
                if selected_column_types.get(matched_column) != "date":
                    raise ValueError(f"Matched column '{matched_column}' is not a date column, but the query requires date processing.")
            else:
                # Ensure the matched column is of type numerical
                if selected_column_types.get(matched_column) != "numerical":
                    raise ValueError(f"Matched column '{matched_column}' is not numerical, but the query requires numerical processing.")

        matched_columns.append(matched_column)

    # Check if all conditions reference the same column
    unique_columns = set(matched_columns)

    # Handle single-column case for numerical or date conditions
    if len(unique_columns) == 1:
        matched_column = unique_columns.pop()

        if selected_column_types.get(matched_column) == "date":
            # Process "not between" date conditions
            not_between_match = re.search(r"not between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", prompt, re.IGNORECASE)
            if not_between_match:
                start_date, end_date = pd.to_datetime(not_between_match.group(1)), pd.to_datetime(not_between_match.group(2))
                filtered_rows = dataset[
                    ~((pd.to_datetime(dataset[matched_column]) >= start_date) & 
                      (pd.to_datetime(dataset[matched_column]) <= end_date))
                ]
                return {
                    "column_names": matched_column,
                    "values": f"not between {start_date.date()} and {end_date.date()}",
                    "row_ids": filtered_rows.index.tolist(),
                }

            # Process "between" date conditions
            range_match = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", prompt, re.IGNORECASE)
            if range_match:
                start_date, end_date = pd.to_datetime(range_match.group(1)), pd.to_datetime(range_match.group(2))
                filtered_rows = dataset[
                    (pd.to_datetime(dataset[matched_column]) >= start_date) & 
                    (pd.to_datetime(dataset[matched_column]) <= end_date)
                ]
                return {
                    "column_names": matched_column,
                    "values": f"between {start_date.date()} and {end_date.date()}",
                    "row_ids": filtered_rows.index.tolist(),
                }

        elif selected_column_types.get(matched_column) == "numerical":
            # Process "not between" numerical conditions
            not_between_match = re.search(r"not between\s+(\d+)\s+and\s+(\d+)", prompt, re.IGNORECASE)
            if not_between_match:
                lower, upper = map(int, not_between_match.groups())
                filtered_rows = dataset[
                    ~((dataset[matched_column] >= lower) & (dataset[matched_column] <= upper))
                ]
                return {
                    "column_names": matched_column,
                    "values": f"not between {lower} and {upper}",
                    "row_ids": filtered_rows.index.tolist(),
                }

            # Process "between" numerical conditions
            range_match = re.search(r"between\s+(\d+)\s+and\s+(\d+)", prompt, re.IGNORECASE)
            if range_match:
                lower, upper = map(int, range_match.groups())
                filtered_rows = dataset[
                    (dataset[matched_column] >= lower) & (dataset[matched_column] <= upper)
                ]
                return {
                    "column_names": matched_column,
                    "values": f"between {lower} and {upper}",
                    "row_ids": filtered_rows.index.tolist(),
                }


    # Process multiple conditions normally
    sub_results = []
    for condition in conditions:
        try:
            # Handle "between" and "not between"
            if "between" in condition.lower():
                negation = "not between" in condition.lower()
                if re.search(r"\d{4}-\d{2}-\d{2}", condition):
                    # Handle date conditions
                    range_match = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", condition, re.IGNORECASE)
                    if range_match:
                        start_date, end_date = pd.to_datetime(range_match.group(1)), pd.to_datetime(range_match.group(2))
                        matched_column = match_column_to_query_with_fuzzy_llm(condition, selected_column_embeddings)

                        if negation:
                            filtered_rows = dataset[
                                ~((pd.to_datetime(dataset[matched_column]) >= start_date) &
                                  (pd.to_datetime(dataset[matched_column]) <= end_date))
                            ]
                        else:
                            filtered_rows = dataset[
                                (pd.to_datetime(dataset[matched_column]) >= start_date) &
                                (pd.to_datetime(dataset[matched_column]) <= end_date)
                            ]

                        sub_results.append({
                            "column_names": matched_column,
                            "values": f"{'not ' if negation else ''}between {start_date.date()} and {end_date.date()}",
                            "row_ids": filtered_rows.index.tolist()
                        })
                        continue
                else:
                    # Handle numerical conditions
                    range_match = re.search(r"between\s+(\d+)\s+and\s+(\d+)", condition, re.IGNORECASE)
                    if range_match:
                        lower, upper = map(int, range_match.groups())
                        matched_column = match_column_to_query_with_fuzzy_llm(condition, selected_column_embeddings)

                        if negation:
                            filtered_rows = dataset[
                                ~((dataset[matched_column] >= lower) & (dataset[matched_column] <= upper))
                            ]
                        else:
                            filtered_rows = dataset[
                                (dataset[matched_column] >= lower) & (dataset[matched_column] <= upper)
                            ]

                        sub_results.append({
                            "column_names": matched_column,
                            "values": f"{'not ' if negation else ''}between {lower} and {upper}",
                            "row_ids": filtered_rows.index.tolist()
                        })
                        continue

            # Delegate other conditions to the query router
            sub_results.append(
                process_query_router_with_fuzzy_matching(condition, dataset, selected_column_embeddings, selected_column_types,tokenizer, model, device)
            )

        except Exception as e:
            print(f"Error processing sub-condition '{condition}': {e}")
            sub_results.append({"column_names": None, "values": None, "row_ids": []})

    # Combine results based on logical operators
    final_row_ids = set(sub_results[0]["row_ids"]) if sub_results else set()
    for i, operator in enumerate(operators):
        next_row_ids = set(sub_results[i + 1]["row_ids"])
        if operator == "and":
            final_row_ids &= next_row_ids
        elif operator == "or":
            final_row_ids |= next_row_ids

    # Construct the final result
    return {
        "column_names": list({res["column_names"] for res in sub_results if res["column_names"]}),
        "values": [res["values"] for res in sub_results if res["values"] is not None],
        "row_ids": list(final_row_ids)
    }


########################## Func 3.1. Handling operations and conditions in the query with multiple conditions
def parse_prompt(prompt):
    """
    Parses a prompt to detect sub-queries and logical operators.

    Parameters:
    - prompt (str): The user's query containing multiple conditions.

    Returns:
    - tuple: A list of conditions and a list of detected logical operators.
    """
    # Define logical operators
    operators = {"and": "and", "or": "or", "not": "not"}
    
    # Handle "not between" and "between" explicitly before splitting
    patterns = [r"not\s+between\s+\d+\s+and\s+\d+", r"between\s+\d+\s+and\s+\d+"]
    special_matches = []
    for pattern in patterns:
        special_matches += re.findall(pattern, prompt, flags=re.IGNORECASE)
    
    # Temporarily replace matches with placeholders
    temp_placeholders = {}
    for idx, match in enumerate(special_matches):
        placeholder = f"<SPECIAL_{idx}>"
        temp_placeholders[placeholder] = match
        prompt = prompt.replace(match, placeholder)

    # Split the prompt into conditions based on logical operators
    conditions = re.split(r"\band\b|\bor\b", prompt, flags=re.IGNORECASE)
    conditions = [condition.strip() for condition in conditions]

    # Replace placeholders back with original matches
    for idx, _ in enumerate(conditions):
        for placeholder, original in temp_placeholders.items():
            conditions[idx] = conditions[idx].replace(placeholder, original)

    # Detect logical operators in the prompt
    operator_matches = re.findall(r"\band\b|\bor\b", prompt, flags=re.IGNORECASE)
    operators_list = [operators[op.lower()] for op in operator_matches]

    return conditions, operators_list


########################## Func. 3.2 Matching a query to the most relevant column
def match_column_to_query_with_fuzzy_llm(query, column_embeddings):
    """
    Matches a query to the most relevant column, handling typos with fuzzy matching.

    Parameters:
    - query (str): The user's query.
    - column_embeddings (dict): Dictionary with selected column names as keys and their embeddings as values.

    Returns:
    - str: The best matching column name.
    """
    # Extract the likely column name from the query
    extracted_column_name = extract_column_name_from_query(query)
    # Extract column names
    column_names = list(column_embeddings.keys())

    # Check for "between" or "not between" conditions
    if "between" in query or "not between" in query:
        # Check for date-related patterns
        if re.search(r"\d{4}-\d{2}-\d{2}", query) or "date" in query.lower():
            date_columns = [col for col in column_names if "date" in col.lower()]
            if date_columns:
                best_match = max(date_columns, key=lambda col: fuzz.ratio(extracted_column_name.lower(), col.lower()))
                return best_match

        # Assume numerical if no date pattern is detected
        numerical_columns = column_names  # Assume all remaining columns are potential numerical columns
        best_match = max(numerical_columns, key=lambda col: fuzz.ratio(extracted_column_name.lower(), col.lower()))
        return best_match

    # Fuzzy matching for all column names (default behavior for other cases)
    best_match = max(column_names, key=lambda col: fuzz.ratio(extracted_column_name.lower(), col.lower()))

    return best_match


############## Func 3.2.1 Extracting the potential column name
def extract_column_name_from_query(query):
    """
    Extracts a potential column name from the query.

    Parameters:
    - query (str): The user's query.

    Returns:
    - str: The extracted substring likely to be the column name.
    """
    # Match phrases like "where columnname equals" or "where columnname is"
    match = re.search(r"where\s+([\w\s]+)\s+(equal|equals|is|contains|is not|isn't|does not)", query, re.IGNORECASE)
    if match:
        return match.group(1).strip()  # Extract the part likely to be the column name
    return query  # Fallback to the whole query if no match

########################## Func 4. (3.3) Handling standard query
def process_query_router_with_fuzzy_matching(query, dataset, selected_column_embeddings, selected_column_types, tokenizer, model, device):
    """
    Routes a query to the appropriate handler using fuzzy column name matching.
    """
    query = preprocess_query(query)

    # Deduce column and value from the query
    deduced = deduce_column_and_value(query, dataset, selected_column_embeddings, tokenizer, model, device)

    matched_column = deduced["column"]
    matched_value = deduced["value"]
    # If no column or value is deduced, fallback to fuzzy matching
    if not matched_column:
        matched_column = match_column_to_query_with_fuzzy_llm(query, selected_column_embeddings)

    if not matched_column:
        return {"error": "No matching column found", "query": query}

    # Detect the query context, considering the matched column
    context = detect_query_context_with_llm(query, selected_column_types, tokenizer, model, device, matched_column)
    # Route the query to the appropriate handler
    try:
        if context == "date":
            result = process_date_query(query, dataset, matched_column, matched_value_deduct = matched_value)
        elif context == "numerical":
            result = process_numerical_query(query, dataset, matched_column, tokenizer, model, device)
        elif context == "categorical":
            result = process_categorical_query(query, dataset, matched_column, matched_value_deduct = matched_value)
        else:
            raise ValueError(f"Unsupported query context: {context}")
    except KeyError as e:
        return {"error": str(e), "query": query, "matched_column": matched_column, "context": context}
    except ValueError as e:
        return {"error": str(e), "query": query, "matched_column": matched_column, "context": context}

    # Include deduced value if applicable
    if matched_value:
        result["deduced_value"] = matched_value

    return result


#### Helper Function
def deduce_column_and_value(query, dataset, selected_column_embeddings, tokenizer, model, device, score_cutoff=50):
    """
    Deduce column names and values dynamically from a query.

    Parameters:
    - query (str): The textual query.
    - dataset (pd.DataFrame): The dataset to reference.
    - selected_column_embeddings (dict): Embeddings of column names for semantic matching.
    - tokenizer (transformers.Tokenizer): Tokenizer for embedding text.
    - model (transformers.Model): Model for generating embeddings.
    - device (str): Device to use for inference.
    - score_cutoff (int): Minimum score for fuzzy matching.

    Returns:
    - dict: A dictionary with deduced column and value.
    """
    query = query.lower().strip()

    # Step 1: Match query to column names
    column_candidates = dataset.columns.tolist()
    column_match = fuzzy_process.extractOne(query, column_candidates, score_cutoff=score_cutoff)
    
    if column_match:
        matched_column = column_match[0]  # Best-matched column
    else:
        # Semantic matching using embeddings if available
        matched_column = None
        max_similarity = 0
        for col in column_candidates:
            if col in selected_column_embeddings:
                similarity = torch.nn.functional.cosine_similarity(
                    embed_texts([query], tokenizer, model, device),
                    selected_column_embeddings[col],
                    dim=1
                ).item()
                if similarity > max_similarity and similarity > 0.5:  # Set a threshold for embeddings
                    max_similarity = similarity
                    matched_column = col

    if not matched_column:
        return {"column": None, "value": None}

    # Step 2: Match query to values in the matched column
    unique_values = dataset[matched_column].dropna().astype(str).tolist()
    value_match = fuzzy_process.extractOne(query, unique_values, score_cutoff=score_cutoff)

    matched_value = value_match[0] if value_match else None

    return {"column": matched_column, "value": matched_value}




########################## Func 4.1 Matching Context
def detect_query_context_with_llm(query, column_types, tokenizer, model, device, matched_column=None):
    """
    Detects the query context using embeddings and fallback rules.

    Parameters:
    - query (str): The input query string.
    - column_types (dict): Dictionary mapping column names to their types.
    - matched_column (str): The matched column name, if already identified.

    Returns:
    - str: Detected context ('date', 'numerical', or 'categorical').
    """
    query = query.lower()
    
    # Check for explicit date-related keywords
    if "between" in query or "not between" in query or re.search(r"\d{4}-\d{2}-\d{2}", query):
        return "date"

    QUERY_CONTEXT_DESCRIPTIONS = {
    "date": "This query involves date-based comparisons like before, after, or on a specific date.",
    "numerical": "This query involves numerical comparisons like greater than, less than, or equal to a value.",
    "categorical": "This query involves finding rows with specific text or category values."
    }

    # Embed the query and context descriptions
    query_embedding = embed_texts([query],tokenizer, model, device)
    context_keys = list(QUERY_CONTEXT_DESCRIPTIONS.keys())
    context_embeddings = embed_texts(context_keys,tokenizer, model, device)

    # Compute similarity scores
    similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, context_embeddings, dim=1)
    best_context_idx = similarity_scores.argmax().item()
    detected_context = context_keys[best_context_idx]

    # Override context if the matched column has a known type
    if matched_column and matched_column in column_types:
        return column_types[matched_column]

    return detected_context



########################## Func 4.2 Handling date queries
def process_date_query(query, dataset, matched_column, matched_value_deduct= None):
    """
    Handles date-based queries for specific dates, ranges, relative dates, and periods.

    Parameters:
    - query (str): The user's query.
    - dataset (pd.DataFrame): The dataset to query.
    - matched_column (str): The matched date column.

    Returns:
    - dict: Query results with column names, values, and matching row IDs.
    """
    query = query.lower()
    today = pd.Timestamp.now()

    # Ensure the column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(dataset[matched_column]):
        try:
            dataset[matched_column] = pd.to_datetime(dataset[matched_column], errors="coerce")
        except Exception as e:
            raise ValueError(f"Failed to convert column '{matched_column}' to datetime: {e}")

    # Drop rows with invalid or missing dates
    valid_dates = dataset[matched_column].dropna()

    if valid_dates.empty:
        raise ValueError(f"No valid dates found in column '{matched_column}'.")
    
    # Handle missing values
    if "is missing" in query or "equals null" in query or "is null" in query:
        filtered_rows = dataset[dataset[matched_column].isnull()]
        return {
            "column_names": matched_column,
            "values": "missing",
            "row_ids": filtered_rows.index.tolist()
        }

    # Handle empty values
    if "equals ''" in query or "equals empty" in query or "is empty" in query:
        filtered_rows = dataset[dataset[matched_column].fillna("").astype(str).str.strip() == ""]
        return {
            "column_names": matched_column,
            "values": "empty",
            "row_ids": filtered_rows.index.tolist()
        }
    
    if matched_value_deduct is not None:
        # Normalize the matched value for comparison
        matched_value_deduct = re.sub(r'[^a-zA-Z0-9]', '', str(matched_value_deduct)).lower()

        # Filter rows where the column matches the deduced value
        filtered_rows = dataset[
            dataset[matched_column].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)).lower() == matched_value_deduct
            )
        ]
        # Return result
        return {
            "column_names": matched_column,
            "values": matched_value_deduct,
            "row_ids": filtered_rows.index.tolist()
        }

    # Min/Max Dates
    if "earliest" in query or "smallest" in query:
        min_date = valid_dates.min()
        filtered_rows = dataset[dataset[matched_column] == min_date]
        return {
            "column_names": matched_column,
            "values": [str(min_date.date())],  # Convert to string
            "row_ids": filtered_rows.index.tolist(),
        }
    elif "latest" in query or "largest" in query:
        max_date = valid_dates.max()
        filtered_rows = dataset[dataset[matched_column] == max_date]
        return {
            "column_names": matched_column,
            "values": [str(max_date.date())],  # Convert to string
            "row_ids": filtered_rows.index.tolist(),
        }

    # Handle "unique values" query
    if "unique values in" in query:
        unique_values = dataset[matched_column].dropna().unique().tolist()
        return {
            "column_names": matched_column,
            "values": unique_values,
            "row_ids": []
        }

    # Absolute Dates: Before/After/On
    date_pattern = r"\d{4}-\d{2}-\d{2}"  # Matches YYYY-MM-DD
    match = re.search(date_pattern, query)
    if match:
        query_date = pd.to_datetime(match.group())
        if "before" in query:
            filtered_rows = dataset[dataset[matched_column] < query_date]
            return {
                "column_names": matched_column,
                "values": [f"Before {query_date.date()}"],
                "row_ids": filtered_rows.index.tolist()
            }
        elif "after" in query:
            filtered_rows = dataset[dataset[matched_column] > query_date]
            return {
                "column_names": matched_column,
                "values": [f"After {query_date.date()}"],
                "row_ids": filtered_rows.index.tolist()
            }
        elif "on" in query:
            filtered_rows = dataset[dataset[matched_column] == query_date]
            return {
                "column_names": matched_column,
                "values": [f"On {query_date.date()}"],
                "row_ids": filtered_rows.index.tolist()
            }

    # Relative Date Queries
    if "within the last" in query:
        days_match = re.search(r"within the last (\d+) days", query)
        if days_match:
            days = int(days_match.group(1))
            start_date = today - pd.Timedelta(days=days)
            filtered_rows = dataset[dataset[matched_column] >= start_date]
            return {
                "column_names": matched_column,
                "values": [f"Within the last {days} days"],
                "row_ids": filtered_rows.index.tolist()
            }
    elif "newer than" in query:
        days_match = re.search(r"newer than (\d+) days", query)
        if days_match:
            days = int(days_match.group(1))
            start_date = today - pd.Timedelta(days=days)
            filtered_rows = dataset[dataset[matched_column] >= start_date]
            return {
                "column_names": matched_column,
                "values": [f"Newer than {days} days"],
                "row_ids": filtered_rows.index.tolist()
            }
    elif "older than" in query:
        days_match = re.search(r"older than (\d+) days", query)
        if days_match:
            days = int(days_match.group(1))
            end_date = today - pd.Timedelta(days=days)
            filtered_rows = dataset[dataset[matched_column] < end_date]
            return {
                "column_names": matched_column,
                "values": [f"Older than {days} days"],
                "row_ids": filtered_rows.index.tolist()
            }

    # Specific Periods
    if "in" in query:
        month_year_match = re.search(r"in (\w+ \d{4})", query)  # Matches 'July 2023'
        if month_year_match:
            period = month_year_match.group(1)
            try:
                start_date = pd.Timestamp(f"1 {period}")
                end_date = start_date + pd.offsets.MonthEnd()
                filtered_rows = dataset[
                    (dataset[matched_column] >= start_date) & (dataset[matched_column] <= end_date)
                ]
                return {
                    "column_names": matched_column,
                    "values": [f"In {period}"],
                    "row_ids": filtered_rows.index.tolist()
                }
            except Exception as e:
                raise ValueError(f"Invalid period format in query: {period} - {str(e)}")

        year_match = re.search(r"in the year (\d{4})", query)  # Matches '2022'
        if year_match:
            year = int(year_match.group(1))
            start_date = pd.Timestamp(f"1 Jan {year}")
            end_date = pd.Timestamp(f"31 Dec {year}")
            filtered_rows = dataset[
                (dataset[matched_column] >= start_date) & (dataset[matched_column] <= end_date)
            ]
            return {
                "column_names": matched_column,
                "values": [f"In the year {year}"],
                "row_ids": filtered_rows.index.tolist()
            }

    if "not between" in query:
        date_match = re.search(r"not between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", query)
        if date_match:
            start_date, end_date = pd.to_datetime(date_match.group(1)), pd.to_datetime(date_match.group(2))
            filtered_rows = dataset[
                ~((dataset[matched_column] >= start_date) & (dataset[matched_column] <= end_date))
            ]
            return {
                "column_names": matched_column,
                "values": [f"Not between {start_date.date()} and {end_date.date()}"],
                "row_ids": filtered_rows.index.tolist()
            }

    # Unhandled query
    raise ValueError(f"Unsupported date query: {query}")




########################## Func 4.3 Handling numerical queries
def process_numerical_query(query, dataset, matched_column,tokenizer, model, device, matched_value_deduct= None):
    """
    Handles numerical queries with enhanced flexibility and embedding-based matching.
    """
    query = query.lower()

    # Match query to closest concept using embeddings
    concept = match_query_to_numerical_concept(query,tokenizer, model, device)



    # Handle missing values
    if "is missing" in query or "equals null" in query or "is null" in query:
        filtered_rows = dataset[dataset[matched_column].isnull()]
        return {
            "column_names": matched_column,
            "values": "missing",
            "row_ids": filtered_rows.index.tolist()
        }

    # Handle empty values
    if "equals ''" in query or "equals empty" in query or "is empty" in query:
        filtered_rows = dataset[dataset[matched_column].fillna("").astype(str).str.strip() == ""]
        return {
            "column_names": matched_column,
            "values": "empty",
            "row_ids": filtered_rows.index.tolist()
        }
    
    if matched_value_deduct is not None:
        # Normalize the matched value for comparison
        matched_value_deduct = re.sub(r'[^a-zA-Z0-9]', '', str(matched_value_deduct)).lower()

        # Filter rows where the column matches the deduced value
        filtered_rows = dataset[
            dataset[matched_column].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)).lower() == matched_value_deduct
            )
        ]
        # Return result
        return {
            "column_names": matched_column,
            "values": matched_value_deduct,
            "row_ids": filtered_rows.index.tolist()
        }
    
    # Handle unique values
    if concept == "unique values":
        unique_values = dataset[matched_column].dropna().unique().tolist()
        return {
            "column_names": matched_column,
            "values": unique_values,
            "row_ids": []
        }


    # Handle extreme values
    if concept in ["highest", "maximum", "largest"]:
        max_value = dataset[matched_column].max()
        filtered_rows = dataset[dataset[matched_column] == max_value]
        return {
            "column_names": matched_column,
            "values": f"highest {max_value}",
            "row_ids": filtered_rows.index.tolist()
        }

    if concept in ["lowest", "minimum"]:
        min_value = dataset[matched_column].min()
        filtered_rows = dataset[dataset[matched_column] == min_value]
        return {
            "column_names": matched_column,
            "values": f"lowest {min_value}",
            "row_ids": filtered_rows.index.tolist()
        }

    # Handle range queries with "between"
    between_match = re.search(r"between (\d+) and (\d+)", query)
    if between_match:
        lower, upper = map(int, between_match.groups())
        filtered_rows = dataset[(dataset[matched_column] >= lower) & (dataset[matched_column] <= upper)]
        return {
            "column_names": matched_column,
            "values": f"between {lower} and {upper}",
            "row_ids": filtered_rows.index.tolist(),
        }

    # Handle range queries like "greater than X and smaller than Y"
    range_match = re.search(r"(greater|higher) than (\d+).*?(smaller|less|below) than (\d+)", query)
    if range_match:
        lower, upper = map(int, range_match.groups()[1::2])
        filtered_rows = dataset[(dataset[matched_column] > lower) & (dataset[matched_column] < upper)]
        return {
            "column_names": matched_column,
            "values": f"range {lower} to {upper}",
            "row_ids": filtered_rows.index.tolist(),
        }

    # Handle simple comparisons
    if "greater" in query or "higher" in query or "more" in query or "above" in query:
        comparison = "greater"
    elif "less" in query or "smaller" in query or "below" in query:
        comparison = "less"
    elif "equal" in query or "equals" in query or re.match(r"^\D*\d+\D*$", query):
        comparison = "equal"  # Default to "equal" if no explicit comparison

    # Fallback to equality if no explicit comparison keyword is found
    if comparison is None or concept == "comparison" and not any(
        keyword in query for keyword in ["greater", "higher", "less", "smaller", "above", "below"]
    ):
        comparison = "equal"

    # Extract numerical value from query
    value_match = re.search(r"\d+", query)
    if not value_match:
        raise ValueError(f"No numerical value found in query: {query}")
    value = int(value_match.group())

    # Apply comparison logic
    if comparison == "greater":
        filtered_rows = dataset[dataset[matched_column] > value]
        return {
            "column_names": matched_column,
            "values": f"greater {value}",
            "row_ids": filtered_rows.index.tolist(),
        }
    elif comparison == "less":
        filtered_rows = dataset[dataset[matched_column] < value]
        return {
            "column_names": matched_column,
            "values": f"less {value}",
            "row_ids": filtered_rows.index.tolist(),
        }
    elif comparison == "equal":
        filtered_rows = dataset[dataset[matched_column] == value]
        return {
            "column_names": matched_column,
            "values": f"equal {value}",
            "row_ids": filtered_rows.index.tolist(),
        }

    # If no pattern matches
    raise ValueError(f"Unsupported numerical condition in query: {query}")

############## Func 4.3.1 Matching query to the closest numerical concept 
def match_query_to_numerical_concept(query, tokenizer, model, device, use_fuzzy=True):
    """
    Matches the query to the closest numerical concept using embeddings and optionally fuzzy matching.

    Parameters:
    - query (str): The user's query.
    - use_fuzzy (bool): Whether to use fuzzy matching as a fallback for typos.

    Returns:
    - str: The closest matching concept.
    """
    # Define key concepts for numerical queries
    numerical_concepts = {
        "greater than": "comparison",
        "higher than": "comparison",
        "more than": "comparison",
        "above": "comparison",
        "smaller than": "comparison",
        "less than": "comparison",
        "below": "comparison",
        "equal to": "comparison",
        "equals": "comparison",
        "highest": "extreme",
        "maximum": "extreme",
        "largest": "extreme",
        "lowest": "extreme",
        "minimum": "extreme",
        "unique values": "unique"
    }

    # Precompute embeddings for these concepts
    concept_embeddings = {key: embed_texts([key],tokenizer, model,device) for key in numerical_concepts.keys()}

    # Embed the query
    query_embedding = embed_texts([query],tokenizer, model,device)

    # Compute similarity scores for embedding-based matching
    similarity_scores = {
        concept: torch.nn.functional.cosine_similarity(query_embedding, embedding, dim=1).item()
        for concept, embedding in concept_embeddings.items()
    }

    # Find the best match based on embeddings
    best_concept = max(similarity_scores, key=similarity_scores.get)

    # If fuzzy matching is enabled and similarity is low, use fuzzy matching as a fallback
    if use_fuzzy:
        best_fuzzy_match, score = fuzzy_process.extractOne(query, list(numerical_concepts.keys()))
        if score > 70:  # Fuzzy matching threshold
            return best_fuzzy_match

    return best_concept



########################## Func 4.4 Handling categorical queries
def process_categorical_query(query, dataset, matched_column, matched_value_deduct=None):
    """
    Processes categorical queries, including handling synonyms, equality,
    negation, substring matching, empty/missing values, logical operators,
    and requests for unique/distinct values.
    """
    
    # Define base conditions and synonyms
    base_conditions = {
        "most_common": ["most common", "most frequent", "most used"],
        "unique_values": ["unique values", "distinct values", "different values", "unique", "distinct"],
        "count": ["its count", "number of occurrences", "frequency"],
        "missing": ["is missing", "equals null", "is null"],
        "empty": ["equals ''", "equals empty", "is empty"],
        "not_equals": [
            "not equals", "is not", "does not equal", "isn't", "doesn't equal", "does not contain",
            "not include", "excludes", "not have", "not has", "doesn't have"
        ],
        "equals": ["equals", "equal", "is", "matches","contains", "includes", "has"],
        
    }

    # Expand conditions with synonyms and variants
    expanded_conditions = {
        base: set(word for term in terms for word in get_synonyms_and_variants(term))
        for base, terms in base_conditions.items()
    }

    # Add original terms for completeness
    for base, terms in base_conditions.items():
        expanded_conditions[base].update(terms)

    # Normalize the query
    query = query.strip().lower()

    # Handle logical operators (and/or)
    if " and " in query or " or " in query:
        logical_match = re.split(r"\s(and|or)\s", query)
        if len(logical_match) >= 3:
            left_query = logical_match[0].strip()
            operator = logical_match[1].strip()
            right_query = logical_match[2].strip()

            # Process left and right subconditions
            left_result = process_categorical_query(left_query, dataset, matched_column)
            right_result = process_categorical_query(right_query, dataset, matched_column)

            # Combine results based on operator
            if operator == "and":
                combined_row_ids = list(set(left_result["row_ids"]).intersection(set(right_result["row_ids"])))
            elif operator == "or":
                combined_row_ids = list(set(left_result["row_ids"]).union(set(right_result["row_ids"])))

            return {
                "column_names": matched_column,
                "values": f"{left_result['values']} {operator} {right_result['values']}",
                "row_ids": combined_row_ids,
            }

    # Special case for unique/distinct queries
    if "unique" in query or "distinct" in query:
        unique_values = dataset[matched_column].dropna().unique().tolist()
        return {
            "column_names": matched_column,
            "values": unique_values,
            "row_ids": []  # No specific rows to highlight
        }

    if matched_value_deduct is not None:
        # Normalize the matched value for comparison
        matched_value_deduct = re.sub(r'[^a-zA-Z0-9]', '', str(matched_value_deduct)).lower()

        # Filter rows where the column matches the deduced value
        filtered_rows = dataset[
            dataset[matched_column].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)).lower() == matched_value_deduct
            )
        ]

        # Return result
        return {
            "column_names": matched_column,
            "values": matched_value_deduct,
            "row_ids": filtered_rows.index.tolist()
        }

    # Match query against conditions
    matched_condition = None
    for base, synonyms in expanded_conditions.items():
        if any(keyword in query for keyword in synonyms):
            matched_condition = base
            break

    if not matched_condition:
        raise ValueError(f"Unsupported or unrecognized condition in query: {query}")

    # Handle "unique values"
    if matched_condition == "unique_values":
        unique_values = dataset[matched_column].dropna().unique().tolist()
        return {
            "column_names": matched_column,
            "values": unique_values,
            "row_ids": []  # No specific rows to highlight
        }

    # Handle "most common"
    if matched_condition == "most_common":
        most_common_value = dataset[matched_column].mode()[0]  # Get the most common value
        filtered_rows = dataset[dataset[matched_column] == most_common_value]
        return {
            "column_names": matched_column,
            "values": most_common_value,
            "row_ids": filtered_rows.index.tolist()
        }

    # Handle "missing" values
    if matched_condition == "missing":
        filtered_rows = dataset[dataset[matched_column].isnull()]
        return {
            "column_names": matched_column,
            "values": "missing",
            "row_ids": filtered_rows.index.tolist()
        }

    # Handle "empty" values
    if matched_condition == "empty":
        filtered_rows = dataset[dataset[matched_column].fillna("").str.strip() == ""]
        return {
            "column_names": matched_column,
            "values": "empty",
            "row_ids": filtered_rows.index.tolist()
        }

    # Handle equality and inequality
    if matched_condition in ["equals", "not_equals"]:
        value_match = re.search(
            r"(not equals|does not equal|doesn't have|is not|isn't|does not contain|not includes|excludes|not have|not has|equals|equal|is|matches|contains|includes|has)\s+['\"]?([\w\s-]*)['\"]?",
            query,
            re.IGNORECASE
        )
        # Adjust regex to be more flexible and case-insensitive
        #value_match = re.search(r"(equals|equal|is|matches|not equals|contains|includes|has|does not contain|not includes|excludes|is not|doesn't have|does not equal|isn't)\s+['\"]?([\w\s-]*)['\"]?", query, re.IGNORECASE)
        
        if not value_match:
            raise ValueError(f"No value found for comparison in query: {query}")
        condition_value = value_match.group(2).strip().lower()  # Normalize condition value for comparison

        # Normalize the query value for comparison
        normalized_condition_value = re.sub(r'[^a-zA-Z0-9]', '', condition_value).lower()

        if matched_condition == "equals":
            # Normalize dataset values for comparison
            filtered_rows = dataset[
                dataset[matched_column].apply(
                    lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)).lower() == normalized_condition_value
                )
            ]
            return {
                "column_names": matched_column,
                "values": condition_value,
                "row_ids": filtered_rows.index.tolist()
            }
        elif matched_condition == "not_equals":
            # Normalize dataset values for comparison
            filtered_rows = dataset[
                dataset[matched_column].apply(
                    lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)).lower() != normalized_condition_value
                )
            ]
            return {
                "column_names": matched_column,
                "values": f"Not {condition_value}",
                "row_ids": filtered_rows.index.tolist()
            }

    # Handle "count"
    if matched_condition == "count":
        value_counts = dataset[matched_column].value_counts().to_dict()
        return {
            "column_names": matched_column,
            "values": value_counts,
            "row_ids": []  # No specific rows to highlight
        }

    # Fallback for unsupported conditions
    raise ValueError(f"Condition type '{matched_condition}' not supported for query: {query}")


############## Func 4.4.1 Checking Synonyms
def get_synonyms_and_variants(word):
    """
    Retrieve synonyms and inflected variants using WordNet and lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    synonyms = set()

    # Get synonyms from WordNet
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())

    # Add base form and inflected forms
    base_form = lemmatizer.lemmatize(word, pos='v')  # Base form for verbs
    synonyms.add(base_form)
    synonyms.add(base_form + 's')  # Handle singular/plural
    synonyms.add(base_form + 'ed')  # Handle past tense
    synonyms.add(base_form + 'ing')  # Handle gerunds

    return synonyms

########################## Saving to Json file ##########################

def save_results_to_file(results, output_file="query_results.json"):
    """
    Saves the query results to a JSON file. Includes error handling and user feedback.

    Parameters:
    - results (list): Query results to save.
    - output_file (str): Name of the output file.
    """
    try:
        # Ensure the results are valid JSON-serializable objects
        if not isinstance(results, list):
            raise ValueError("Results must be a list.")
        
        # Write results to the specified file
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Confirm successful save
        print(f"\nResults successfully saved to '{os.path.abspath(output_file)}'.")
    
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except IOError as e:
        print(f"IO Error: Unable to save results to file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




























