function model = relaxed_hierarchy_train(Y, X, config)

%% train internal nodes
model = train_internal_nodes(Y, X, config);

%% TODO: set a check point

%% train the leaf nodes
%matlabpool(3);
model = train_leaf_nodes(model, Y, X, config);
%matlabpool close;
