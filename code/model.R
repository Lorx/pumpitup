### CHARGEMENT DES PACKAGES ####

source('C:/Users/Lewis/Desktop/datasciencechallenge/pumpitup/code/eda.R')

library(h2o)

# launching H20 with a maximum of 4 gb Ram

localH2O <<- h2o.init(ip =  "localhost", port = 54321, 
                      nthreads = -1, max_mem_size = "10g")



y <- "status_group"

x <- setdiff(names(df), y)


BM <- h2o.splitFrame(as.h2o(df), ratios = 0.75)

BM_train <- BM[[1]]
BM_test <- BM[[2]]

#### MODELLING ####

AML <- h2o.automl(
  x = x,
  y = y,
  training_frame = BM_train,
  validation_frame = BM_test,
  leaderboard_frame = BM_test,
  nfolds = 5,
  fold_column = NULL,
  weights_column = NULL,
  balance_classes = FALSE,
  class_sampling_factors = NULL,
  max_after_balance_size = 5,
  max_runtime_secs = 3600,
  max_models = NULL,
  stopping_metric = "AUTO",
  #   c(
  #   "AUTO",
  #   "deviance",
  #   "logloss",
  #   "MSE",
  #   "RMSE",
  #   "MAE",
  #   "RMSLE",
  #   "AUC",
  #   "lift_top_group",
  #   "misclassification",
  #   "mean_per_class_error"
  # ),
  stopping_tolerance = NULL,
  stopping_rounds = 3,
  seed = NULL,
  project_name = "AUTOML_PIP",
  exclude_algos =  c("GLM", "DeepLearning", "StackedEnsemble"),
  keep_cross_validation_predictions = FALSE,
  keep_cross_validation_models = FALSE,
  keep_cross_validation_fold_assignment = FALSE,
  sort_metric = "AUTO",
  #   c(
  #   "AUTO",
  #   "deviance",
  #   "logloss",
  #   "MSE",
  #   "RMSE",
  #   "MAE",
  #   "RMSLE",
  #   "AUC",
  #   "mean_per_class_error"
  # ),
  export_checkpoints_dir = NULL
)



#### XGBOOST ####

h2o.xgboost(x = x,
            y = y,
            training_frame = BM_train,
            model_id = "G_Xgboost_10_06_2019" ,
            validation_frame = BM_test,
            nfolds = 10,
            keep_cross_validation_models = TRUE,
            keep_cross_validation_predictions = FALSE,
            keep_cross_validation_fold_assignment = FALSE,
            score_each_iteration = FALSE,
            fold_assignment = c("AUTO", "Random", "Modulo", "Stratified"),
            fold_column = NULL,
            ignore_const_cols = TRUE,
            offset_column = NULL,
            weights_column = NULL,
            stopping_rounds = 0,
            stopping_metric = c("AUTO", "deviance", "logloss", "MSE", "RMSE",
                                "MAE", "RMSLE", "AUC", "lift_top_group", "misclassification",
                                "mean_per_class_error", "custom", "custom_increasing"),
            stopping_tolerance = 0.001,
            max_runtime_secs = 0,
            seed = -1,
            distribution = c("AUTO", "bernoulli", "multinomial", "gaussian",
                             "poisson", "gamma", "tweedie", "laplace", "quantile", "huber"),
            tweedie_power = 1.5,
            categorical_encoding = c("AUTO", "Enum", "OneHotInternal", "OneHotExplicit", "Binary", "Eigen", "LabelEncoder",
                                                          "SortByResponse", "EnumLimited"),
            quiet_mode = TRUE,
            export_checkpoints_dir = NULL,
            ntrees = 50,
            max_depth = 6,
            min_rows = 1,
            min_child_weight = 1,
            learn_rate = 0.3,
            eta = 0.3,
            sample_rate = 1,
            subsample = 1,
            col_sample_rate = 1,
            colsample_bylevel = 1,
            col_sample_rate_per_tree = 1,
            colsample_bytree = 1,
            max_abs_leafnode_pred = 0,
            max_delta_step = 0,
            monotone_constraints = NULL,
            score_tree_interval = 0,
            min_split_improvement = 0,
            gamma = 0,
            nthread = -1,
            max_bins = 256,
            max_leaves = 0,
            min_sum_hessian_in_leaf = 100,
            min_data_in_leaf = 0,
            sample_type = c("uniform", "weighted"),
            normalize_type = c("tree", "forest"),
            rate_drop = 0,
            one_drop = FALSE,
            skip_drop = 0,
            tree_method = c("auto", "exact", "approx", "hist"),
            grow_policy = c("depthwise", "lossguide"),
            booster = c("gbtree","gblinear", "dart"),
            reg_lambda = 1,
            reg_alpha = 0,
            dmatrix_type = c("auto", "dense", "sparse"),
            backend = c("auto", "gpu", "cpu"),
            gpu_id = 0,
            verbose = FALSE)




#### Random forest ####

rf1 <- h2o.randomForest(

  x = x,
  y = y,

  training_frame = BM_train,
  validation_frame = BM_test,

  model_id = "RF_GENERALI",
  nfolds = 10,
  keep_cross_validation_models = TRUE,
  keep_cross_validation_predictions = FALSE,
  keep_cross_validation_fold_assignment = FALSE,
  score_each_iteration = FALSE,
  score_tree_interval = 0,
  fold_assignment = "AUTO",
  fold_column = NULL,
  ignore_const_cols = FALSE,
  # offset_column = NULL,
  weights_column = NULL,
  balance_classes = FALSE,
  class_sampling_factors = NULL,
  max_after_balance_size = NULL,
  max_hit_ratio_k = 0,
  ntrees = 300,
  max_depth = 100,
  min_rows = 10,
  nbins = 10,
  nbins_top_level = 1024,
  nbins_cats = 1024,
  # r2_stopping = Inf,
  stopping_rounds = 0,
  stopping_metric = "AUTO",
  stopping_tolerance = 0.001,
  max_runtime_secs = 0,
  seed = 5,
  build_tree_one_node = FALSE,
  mtries = -1,
  sample_rate = 0.8, #
  sample_rate_per_class = NULL,
  binomial_double_trees = FALSE,
  checkpoint = NULL,
  col_sample_rate_change_per_level = 1,
  col_sample_rate_per_tree = 0.8, #
  min_split_improvement = 1e-05,
  histogram_type = "AUTO",
  categorical_encoding = "AUTO",
  calibrate_model = FALSE,
  calibration_frame = NULL,
  # distribution = "AUTO",
  custom_metric_func = NULL,
  export_checkpoints_dir = NULL,
  verbose = FALSE)

#### GBM ####

GBM <- h2o.gbm(x = x,
               y = y,

               training_frame = BM_train,
               validation_frame = BM_test,

               model_id = "GBM_GENERALI",

               nfolds = 10,
               keep_cross_validation_models = TRUE,
               keep_cross_validation_predictions = FALSE,
               keep_cross_validation_fold_assignment = FALSE,
               score_each_iteration = FALSE,
               score_tree_interval = 0,
               fold_assignment = "AUTO",
               fold_column = NULL,
               ignore_const_cols = FALSE,
               offset_column = NULL,
               weights_column = NULL,
               balance_classes = FALSE,
               class_sampling_factors = NULL,
               max_after_balance_size = 5,
               max_hit_ratio_k = 5,
               ntrees = 1500,
               max_depth = 900,
               min_rows = 15,
               nbins = 20,
               nbins_top_level = 1024,
               nbins_cats = 1024,
               # r2_stopping = Inf,
               stopping_rounds = 0,
               stopping_metric = "AUC",
               stopping_tolerance = 0.001,
               max_runtime_secs = 0,
               seed = 5,
               build_tree_one_node = FALSE,
               learn_rate = 0.005,
               learn_rate_annealing = 0.999,
               distribution = "AUTO",
               quantile_alpha = 0.5,
               tweedie_power = 1.5,
               huber_alpha = 0.9,
               checkpoint = NULL,
               sample_rate = 0.8,
               sample_rate_per_class = NULL,
               col_sample_rate = 0.8,
               col_sample_rate_change_per_level = 1,
               col_sample_rate_per_tree = 0.9,
               min_split_improvement = 1e-05,
               histogram_type = "AUTO",
               max_abs_leafnode_pred = Inf,
               pred_noise_bandwidth = 0,
               categorical_encoding = "AUTO",
               calibrate_model = FALSE,
               calibration_frame = NULL,
               custom_metric_func = NULL,
               export_checkpoints_dir = NULL,
               monotone_constraints = NULL,
               verbose = FALSE)








#### MODEL PERFORMANCE ####

perf <- h2o.performance(bestMo, newdata = BM_test)



mc <- h2o.confusionMatrix(bestMo,
                          thresholds = NULL, metrics = NULL, valid = T)

h2o.confusionMatrix(rf1,
                    thresholds = NULL, metrics = NULL, valid = T)

h2o.confusionMatrix(GBM,
                    thresholds = NULL, metrics = NULL, valid = T)


h2o.accuracy(perf)

h2o.tpr(perf)

h2o.auc(bestMo)


h2o.auc(bestMo, train = T, valid = T, xval = T)

h2o.auc(rf1, train = T, valid = T, xval = T)

h2o.auc(GBM, train = T, valid = T, xval = T)


h2o.giniCoef(bestMo, train = T, valid = T, xval = T)

h2o.giniCoef(rf1, train = T, valid = T, xval = T)

h2o.giniCoef(GBM, train = T, valid = T, xval = T)

# Score History

SH <- h2o.scoreHistory(bestMo)


# Variables importances

var.imp <- h2o.varimp(bestMo)

var.imp_rf <- h2o.varimp(rf1)

h2o.varimp(GBM)



#### PREDICTION ####

 pred <- h2o.predict(rf1, as.h2o( X_test_c1))

pred_GBM <- h2o.predict(GBM, as.h2o( X_test_c1))


Submit <- cbind.data.frame(as.data.frame(pred), X_test_c1)

Submit_GBM <- cbind.data.frame(as.data.frame(pred_GBM), X_test_c1)

Submit_RF_10_06 <- Submit[, c(5, 1)] %>% rename( target = predict)

Submit_RF_10_06 <- Submit[, c(5, 1)] %>% rename( target = predict)

write.table(Submit_RF_10_06, "Submit_RF_10_06.csv", sep = ",", row.names = F, quote = F)

write.table(Submit_RF_10_06, "Submit_RF_10_06.csv", sep = ",", row.names = F, quote = F)
