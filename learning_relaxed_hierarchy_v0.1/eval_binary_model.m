function resp = eval_binary_model(X, model)

resp = X*model.w - model.rho;
