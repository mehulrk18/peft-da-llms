
def calculate_cappa_top1(eval1_df, eval2_df):
    from sklearn.metrics import cohen_kappa_score
    best_1, best_2 = [], []
    eval1_df = eval1_df.drop(columns=["row"])
    eval2_df = eval2_df.drop(columns=["row"])

    for i in range(len(eval1_df)):
        # idxmin() returns the column name (i.e., the model) with the smallest value (rank=1)
        best_1.append(eval1_df.iloc[i].idxmin())
        best_2.append(eval2_df.iloc[i].idxmin())

    # best_annotator1.append(best_model_1)
    # best_annotator2.append(best_model_2)

    # Calculate simple "top-1" agreement
    same_winner_count1 = sum(1 for j in range(len(best_1)) if best_1[j] == best_2[j])
    top1_agreement12 = same_winner_count1 / len(eval1_df)

    # Calculate Cohen’s kappa on the best-model labels
    kappa12 = cohen_kappa_score(best_1, best_2)

    print(f"Top-1 agreement 12: {top1_agreement12:.3f}")
    print(f"Cohen’s kappa on best-model label 12: {kappa12:.3f}\n\n")

    def compute_top_one_percentages(df):
        top_two = {}
        for model in df.columns:
            # Calculate the percentage where the model's rank is 1 or 2.
            percentage = (df[model] <= 1).mean() * 100  # .mean() gives fraction; *100 converts to %
            top_two[model] = percentage
        return top_two

    # Compute percentages for each annotator
    top_two_annotator1 = compute_top_one_percentages(eval1_df)
    top_two_annotator2 = compute_top_one_percentages(eval2_df)

    # Display the computed percentages
    a1, a2 = {}, {}
    print("Annotator 1 top-one percentages:")
    for model, perc in top_two_annotator1.items():
        print(f"  {model}: {perc:.1f}%")
        a1[model] = f"{perc:.1f}%"

    print("\nAnnotator 2 top-one percentages:")
    for model, perc in top_two_annotator2.items():
        print(f"  {model}: {perc:.1f}%")
        a2[model] = f"{perc:.1f}%"

    return {
        "top1_agreement": top1_agreement12,
        "kappa": kappa12,
        "eval1": a1,
        "eval2": a2
    }


if __name__ == '__main__':
    import pandas as pd
    import glob

    files = glob.glob("../../humaneval/*_evaluated.xlsx")

    dataset_wise_files = {
        "scientific": [],
        "medical": [],
        "legal": [],
        "news": []
    }
    eval_scores = {
        "scientific": {},
        "medical": {},
        "legal": {},
        "news": {}
    }

    for file in files:
        if "scientific" in file:
            dataset_wise_files["scientific"].append(file)
        elif "medical" in file:
            dataset_wise_files["medical"].append(file)
        elif "legal" in file:
            dataset_wise_files["legal"].append(file)
        elif "news" in file:
            dataset_wise_files["news"].append(file)

    for dataset, files in dataset_wise_files:
        eval1_df = pd.read_excel(files[0])
        eval2_df = pd.read_excel(files[1])
        eval_scores[dataset] = calculate_cappa_top1(eval1_df, eval2_df)

    print(eval_scores)