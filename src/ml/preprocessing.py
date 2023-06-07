from sklearn.impute import SimpleImputer

# Preprocessing with string "n/a"
# X = [["n/a", 3, 5], [2, "n/a", 9], [8, 7, "n/a"]]
# imp_mean = SimpleImputer(missing_values="n/a", strategy='mean')
# X_new = imp_mean.fit_transform(X)
# print(X_new)

# Preprocessing with numeric data type
X = [[0, 3, 5], [2, 0, 9], [8, 7, 0]]
imp_mean = SimpleImputer(missing_values="n/a", strategy='mean')
X_new = imp_mean.fit_transform(X)
print("Expected Output:")
print([5, 3, 5], [2, 5, 9], [8, 7, 7])
print("Actual Output:")
print(X_new)
