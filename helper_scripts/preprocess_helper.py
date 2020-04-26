import pandas as pd

def read_csv(path, num_examples):

    df = pd.read_csv(path)
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['input_sequence', 'output_sequence']]
    assert len(df.columns) == 2, 'column names should be input_sequence and output_sequence'
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'

    return (df["input_sequence"].values, df["output_sequence"].values)