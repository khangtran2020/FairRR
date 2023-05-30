from sklearn.linear_model import LogisticRegression
def run_sklearn(args, tr_info, va_info, te_info, name, device):
    tr_df = tr_info
    te_df = te_info

    X = tr_df[args.feature].values
    y = tr_df[args.target].values
    X_test = te_df[args.feature].values
    y_test = te_df[args.target].values
    clf = LogisticRegression(max_iter=10000, random_state=0).fit(X, y)
    print("Test acc:", clf.score(X_test, y_test))