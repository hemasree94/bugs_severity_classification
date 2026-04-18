import pandas as pd

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def remove_columns(df, columns_to_remove):
    df = df.drop(columns=columns_to_remove)
    return df

def clean_text(df, text_column):
    df[text_column] = df[text_column].str.lower()
    df[text_column] = df[text_column].str.replace(r'[^\w\s]', '', regex=True)
    return df

def label_encode(df, column_to_encode, label_map):
    df[column_to_encode] = df[column_to_encode].map(label_map)
    return df

def preprocess_data(file_path, columns_to_remove, text_column, column_to_encode, label_map):
    df = read_data(file_path)
    df = remove_columns(df, columns_to_remove)
    df = clean_text(df, text_column)
    df = label_encode(df, column_to_encode, label_map)
    return df

#if __name__ == "__main__":
 #   file_path = "data/raw/bugs.csv"
 #   columns_to_remove = ["id", "raw_severity", "priority", "product", "component", "status", "resolution", "creation_time", "comment_count", "scraped_at"]
  #  text_column = "summary"
   # column_to_encode = "severity"
    #label_map = {
     #   "trivial": 0,
      #  "minor": 1,
      #  "normal": 2,
       # "major": 3,
        #"critical": 4,
    #}
    #preprocessed_df = preprocess_data(file_path, columns_to_remove, text_column, column_to_encode, label_map)
    #preprocessed_df.to_csv("data/preprocessed_bugs.csv", index=False)




