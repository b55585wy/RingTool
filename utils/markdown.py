from typing import Dict, List


def format_results_to_markdown(results_list: List[Dict]) -> str:
    """Formats a list of result dictionaries into a Markdown string."""
    if not results_list:
        return "**Results:**\nNo results available."

    markdown_parts = ["**Results:**"]

    for i, results_dict in enumerate(results_list):
        # Make a copy to avoid modifying the original dict
        current_results = results_dict.copy()
        # Attempt to get task name, default if not present
        task_name = current_results.pop("task", f"Task {i+1}")

        table_header = f"*{task_name}:*\n| Metric         | Value        |\n|----------------|--------------|"
        table_rows = []

        # Define a preferred order for metrics, others will be appended alphabetically
        preferred_order = [
            'mae', 'mae_with_std', 'rmse', 'rmse_with_std',
            'mape', 'mape_with_std', 'pearson', 'pearson_with_std',
            'loss', 'sample_len'
        ]
        # Get keys in preferred order, then add remaining keys sorted alphabetically
        sorted_keys = [key for key in preferred_order if key in current_results] + \
                        sorted([key for key in current_results if key not in preferred_order])

        for key in sorted_keys:
            value = current_results[key]
            # Format floats nicely, convert others to string
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            # Use backticks for value to prevent potential markdown issues and improve alignment
            table_rows.append(f"| {key:<14} | `{value_str}`{' ' * (12 - len(value_str))}|") # Pad key and value within backticks

        markdown_parts.append(table_header + "\n" + "\n".join(table_rows))

    return "\n\n".join(markdown_parts) # Join task tables with double newline
