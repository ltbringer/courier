# Courier

Inspired by the need to get a bunch of excel sheets emailed to me (and which need some preprocessing) 
to not bother me from doing other useful things.


## Clunky stuff
I have used [Gmail API](https://developers.google.com/gmail/api/quickstart/python) to 
automate the workflow. Here are some snippets to work out something different.

```python
maybe_messages = service.users().messages().list(
    userId="me", 
    q="""
    label:<label> 
    from:<email> 
    after:2020/5/1 
    before:2020/5/18 
    has:attachment
    """
).execute()

message_ids = [message["id"] for message in 
                maybe_messages.get("messages")]

maybe_email = service.users().messages().get(
    userId="me", 
    id=message_ids[0])
    .execute() 

maybe_attchments = [part["body"]["attachmentId"]
                    for part in maybe_email["payload"]["parts"] 
                    if part["body"].get("attachmentId")]

attachment = service.users().messages().attachments()
    .get(userId="me", 
        messageId=message_ids[0], 
        id=maybe_attchments[0])
    .execute() 
```
