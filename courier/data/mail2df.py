import pickle
import os.path
import base64
import zipfile
from typing import List, Dict, Tuple
from datetime import datetime

import psutil
import pandas as pd
from pandas.core.frame import DataFrame
from googleapiclient import discovery
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from tqdm import tqdm

from courier.utils.logger import L


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MEMORY_CAP_CRIT = 50


def create_mail_service() -> discovery.Resource:
    """
    Create API instance for gmail.
    returns: googleapiclient.discovery.Resource
    """
    credentials = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    L.debug("Checking for token.")
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            L.info("Refreshed credentials.")
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            credentials = flow.run_local_server(port=0)
            L.info("First time login?")
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
        L.info("Saved credentials in token.pickle.")
        L.info("Gmail API ready to use.")
    return build('gmail', 'v1', credentials=credentials)


def get_mail_ids_with_data(service: discovery.Resource, query: str) -> List[str]:
    """
    Get emails which satisfy the gmail query.

    Refer to readme to get an example of query value.
    Shape of maybe_emails in an ideal scenario, the `key` messages
    is an empty array if there is nothing that matches the query.
    {
        "messages": [{
            "id": str,
            "threadId": str
        }],
        "resultSizeEstimate": int
    }
    """
    maybe_emails = service.users().messages().list(
        userId="me",
        q=query
    ).execute()
    emails = maybe_emails.get("messages")
    L.info("Fetched a list of emails with relevant query = %s.", query)
    return [email["id"] for email in emails]


def get_message_attachments(service: discovery.Resource, message_id: str) -> Dict[str, List[str]]:
    """
    Return a list of attachment-ids in emails referenced by message_id.

    form of response of maybe_message in ideal case, just keys:

    maybe_message.keys()
    > dict_keys(['id', 'threadId', 'labelIds', 'snippet', 'payload', 'sizeEstimate', 'historyId', 'internalDate'])

    maybe_message["payload"].keys()
    > dict_keys(['partId', 'mimeType', 'filename', 'headers', 'body', 'parts'])

    maybe_message["payload"]["parts"][-1].keys()
    > dict_keys(['partId', 'mimeType', 'filename', 'headers', 'body'])

    maybe_message["payload"]["parts"][-1]["body"].keys()
    > dict_keys(['attachmentId', 'size'])
    """
    maybe_message = service.users().messages().get(
        userId="me",
        id=message_id) \
        .execute()
    message_parts = maybe_message.get("payload", {}).get("parts", [])
    return {message_id: [part["body"]["attachmentId"]
            for part in message_parts if part.get("body", {}).get("attachmentId")]}


def get_attachment_ids_from_emails(service: discovery.Resource, email_ids: List[str]):
    """
    Return a flat list of attachments that can be downloaded.
    """
    L.info("Checking emails that contain attachments, "
           "mapping `message_id` of the email to `attachment_id`s associated.")
    return [get_message_attachments(service, email_id) for email_id in email_ids]


def read_xlsx_as_df(b64_data: str) -> DataFrame:
    """
    Returns a
    """
    xlsx_data = base64.urlsafe_b64decode(b64_data.encode("UTF-8"))
    try:
        return pd.read_excel(xlsx_data)
    except zipfile.BadZipFile:
        L.info("Looks like that's not a valid attachment!")


def log_file(date_string: str, query: str):
    with open("journal.log", "w") as f:
        f.write(f"{date_string} - used with query q={query}")


def attachments_as_df(service: discovery.Resource,
                      attachment_data: Dict[str, List[str]]) -> Tuple[List[DataFrame], bool]:
    """
    Save attachment data (a b64 encoded string) to csv.

    maybe_attachment.keys()
    > dict_keys(['size', 'data'])

    maybe_attachment["data"] is the downloadable content.
    """
    data_frames = []
    process = psutil.Process(os.getpgid())
    missing_files_warn = False
    L.info("Load all attachments as df into memory.")
    for email_id, attachment_ids in tqdm(attachment_data.items()):
        for attachment_id in attachment_ids:
            maybe_attachment = service.users().messages().attachments().get(
                userId="me",
                messageId=email_id,
                id=attachment_id
            ).execute()
            if "data" not in maybe_attachment:
                continue
            df = read_xlsx_as_df(maybe_attachment["data"])
            L.info("%0.2f", process.memory_percent())

            if process.memory_percent() > MEMORY_CAP_CRIT:
                L.warning("[CRITICAL]: Memory capacity | Saving files on disk instead.")
                missing_files_warn = True
                date_string = datetime.now().strftime('%d-%M-%Y_%HH-%MM')
                file_name = f"{date_string}.csv"
                df.to_csv(file_name), missing_files_warn
            else:
                data_frames.append(df)
    L.info("Number of data_frames loaded into memory %d", len(data_frames))
    return data_frames, missing_files_warn
