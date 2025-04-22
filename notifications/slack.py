import logging
from typing import Dict, List

import toml
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def format_results_to_slack_blocks(result: Dict) -> List[Dict]:
    """Formats a list of result dictionaries into Slack message blocks."""

    return [
        {
            "text": {
                "emoji": True,
                "text": "📊 Model Evaluation Results",
                "type": "plain_text",
            },
            "type": "header",
        },
        {
            "fields": [
                {"text": "*Metric*", "type": "mrkdwn"},
                {"text": "*Value*", "type": "mrkdwn"},
            ],
            "type": "section",
        },
        {"type": "divider"},
        {
            "fields": [
                {"text": "MAE", "type": "mrkdwn"},
                {"text": f"{result['mae_with_std']}", "type": "mrkdwn"},
            ],
            "type": "section",
        },
        {"type": "divider"},
        {
            "fields": [
                {"text": "RMSE", "type": "mrkdwn"},
                {"text": f"{result['rmse_with_std']}", "type": "mrkdwn"},
            ],
            "type": "section",
        },
        {"type": "divider"},
        {
            "fields": [
                {"text": "MAPE", "type": "mrkdwn"},
                {"text": f"{result['mape_with_std']}", "type": "mrkdwn"},
            ],
            "type": "section",
        },
        {"type": "divider"},
        {
            "fields": [
                {"text": "Pearson", "type": "mrkdwn"},
                {"text": f"{result['pearson_with_std']}", "type": "mrkdwn"},
            ],
            "type": "section",
        },
        {"type": "divider"},
        {
            "fields": [
                {"text": "Loss", "type": "mrkdwn"},
                {"text": f"{result['loss']}", "type": "mrkdwn"},
            ],
            "type": "section",
        },
        {"type": "divider"},
        {
            "fields": [
                {"text": "Sample Length", "type": "mrkdwn"},
                {"text": f"{result['sample_len']}", "type": "mrkdwn"},
            ],
            "type": "section",
        },
        {"type": "divider"},
    ]


def setup_slack(config_toml_path="notifications/slack_configs.toml") -> WebClient:
    # Define the path to the config file relative to the current script
    slack_token = None

    try:
        # Read the TOML configuration file
        config = toml.load(config_toml_path)
        # Extract the token from the [slack] section
        slack_token = config.get("slack", {}).get("token")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_toml_path}")
        # Handle the error appropriately, e.g., exit or raise
        raise SystemExit(f"Configuration file not found: {config_toml_path}")
    except Exception as e:
        logging.error(f"Error reading or parsing configuration file: {e}")
        # Handle the error appropriately
        raise SystemExit(f"Error processing configuration file: {e}")

    if not slack_token:
        logging.error(
            "Error: Slack token not found in the 'slack' section of the configuration file."
        )
        # Handle the missing token appropriately
        raise SystemExit("Slack token missing in configuration.")

    return WebClient(token=slack_token)


def send_slack_message(client: WebClient, channel: str, message: str, blocks=None):
    try:
        # Call the Slack API to send the message
        response = client.chat_postMessage(channel=channel, text=message, blocks=blocks)
        logging.info(f"Message sent to {channel}: {response['message']['text']}")
    except SlackApiError as e:
        logging.error(f"Error sending message: {e.response['error']}")
