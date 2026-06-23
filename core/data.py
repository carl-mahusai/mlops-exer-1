import pandas as pd
import argparse

def convert_to_df(file_path):
    try:
        return pd.read_csv(file_path, encoding_errors='ignore') # or pd.read_excel(file_path)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not read file into DataFrame: {e}")
    
def read_df(args):

    df = pd.DataFrame()

    if args.data:
        df = convert_to_df(args.data)
        label_column = ''
        message_column = ''

        if args.name_of_label_column:
            label_column = args.name_of_label_column

        if args.name_of_message_column:
            message_column = args.name_of_message_column

        if (len(label_column) > 0 and len(message_column) > 0):

            df = df[[label_column, message_column]]

            df = df.rename(columns={label_column: "label", message_column: "message"})

    return df