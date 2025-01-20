import glob

import pandas as pd


if __name__ == "__main__":
    # files = glob.glob("summaries/summaries_*150samples.csv")
    # files.extend(glob.glob("summaries/summaries_*150samples.xlsx"))
    # files.extend(glob.glob("summaries/summaries_unseen_test_*.csv"))
    # files.extend(glob.glob("summaries/summaries_unseen_test_*.xlsx"))

    # for legal
    files = ["summaries/summaries_legal_multilex_150samples.xlsx", "summaries/summaries_legal_eurlex_150samples.xlsx"]
    for file in files:
        samples = 25 if "unseen_test" in file else 50
        print("\nSelecting {} Random Samples from {}".format(samples, file))

        if file.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        # Select 25 random rows from the DataFrame
        random_rows = df.sample(n=samples, random_state=42)  # Set random_state for reproducibility
        if "unseen_test" in file:
            if file.endswith(".xlsx"):
                output_file = file.replace(".xlsx", "_{}samples.xlsx".format(samples))
            else:
                output_file = file.replace(".csv", "_{}samples.csv".format(samples))
        else:
            if file.endswith(".xlsx"):
                output_file = file.replace("150samples.xlsx", "{}samples.xlsx".format(samples))
            else:
                output_file = file.replace("150samples.csv", "{}samples.csv".format(samples))
        # Save the random rows to a new CSV file
        random_rows.to_csv(output_file, index=False)
        print("Random {} samples from {} saved to: {}".format(samples, file, output_file))
