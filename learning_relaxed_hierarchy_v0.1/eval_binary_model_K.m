function resp = eval_binary_model_K(K, model, varargin)

if size(varargin, 2) == 0
    resp = (model.sv_coef' * K(model.SVs, :) - model.rho)';
else
   b_use_global_index = varargin{1};
   if b_use_global_index
     resp = (model.sv_coef' * K(model.global_SV_indice, :) - model.rho)';   
   else
     resp = (model.sv_coef' * K(model.SVs, :) - model.rho)';   
   end
end