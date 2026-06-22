import torch
from spam_checker.util import build_token_list

def predict_sms(
    text,
    model,
    vocab,
    max_length=50,
):
    model.eval()
    tokens = build_token_list(text)
    # tokens = str(text).lower().split()

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    ids = ids[:max_length]

    if len(ids) < max_length:
        ids.extend(
            [vocab["<PAD>"]]
            * (max_length - len(ids))
        )

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=model.device,
    )

    with torch.no_grad():
        logits = model(x)

        prob_spam = torch.sigmoid(
            logits
        ).item()

    prediction = (
        "spam"
        if prob_spam >= 0.5
        else "ham"
    )

    return {
        "prediction": prediction,
        "spam_probability": prob_spam,
    }