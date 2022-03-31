import logging
import sys
from datetime import datetime
from os.path import exists
from autosklearn.experimental.askl2 import AutoSklearnClassifier
from autosklearn.classification import AutoSklearnClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
from joblib import dump, load
from matplotlib.colors import ListedColormap
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='../../logs/model/predict_task_failures.log',
                    level=logging.INFO, format='%(asctime)s %(message)s')


class TraceTaskFailurePredictor:
    """ class for binary classification of task executions in the Google 2011 cluster trace """

    def __init__(self, trace: pd.DataFrame, bool_cols: list, categ_cols: list, str_categ_cols: list,
                 num_cols: list, model_path: str, test_size_frac: float, output_path: str, fit_time: int,
                 run_time: float, memory_limit: int = 3072):
        self.trace = trace
        self.bool_cols = bool_cols
        self.categ_cols = categ_cols
        self.str_categ_cols = str_categ_cols
        self.num_cols = num_cols
        self.model_path = model_path
        self.feature_cols = categ_cols + num_cols + bool_cols  # str_categ_cols are not in here bc it would be duplicate
        self.test_size_frac = test_size_frac
        self.output_path = output_path
        self.fit_time = fit_time
        self.run_time = run_time
        self.memory_limit = memory_limit

    def run(self):
        """ main function """
        logging.info(f"\n ======= START PREDICTOR RUN ======= \n"
                     f"Test size fraction: {self.test_size_frac} \n"
                     f"Automl time: {self.fit_time}")
        X_train, X_test, y_train, y_test = self.preprocess_data()
        model, fit_duration = self.fit_the_data(x_train=X_train, y_train=y_train)
        y_predict, predict_duration = self.predict_the_data(model=model, x_test=X_test)
        eval_df = self.evaluate(model=model, x_test=X_test, y_test=y_test, y_predict=y_predict)
        self.write_eval_df(eval_df=eval_df, path=OUTPUT_PATH)

        logging.info("\n ======= FINISH PREDICTOR RUN =======")

    def evaluate(self, model: AutoSklearnClassifier, x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 y_predict: np.ndarray) -> pd.DataFrame:
        logging.info("Start evaluation")
        if self.fit_time <= 1700:
            logging.info("plotting the performance is skipped because of a small training time")
        else:
            self.plot_performance(model=model)
            pass
        # self.log_eval_metrics(y_test=y_test, y_predict=y_predict, model=model)

        eval_df = self.get_eval_df(x_test=x_test, y_test=y_test, y_predict=y_predict)
        # eval_data = eval_df[["cpu_request", "scheduling_class", "labels", "predictions"]].to_numpy()

        # self.mesh_plot_knn_model(eval_data=eval_data)
        # SMH this doesnt work
        # self.permut_importance_plot(model=model, x_test=x_test, y_test=y_test)
        results = pd.DataFrame.from_dict(model.cv_results_)
        logging.info(results)
        logging.info("Finish evaluation")
        return eval_df

    def preprocess_data(self) -> tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]):
        logging.info("Start preprocessing the data")
        X, y = self.create_x_y(clean_trace=self.trace)
        X = self.cast_and_fill_features(x=X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size_frac, random_state=SAMPLE_SEED
        )
        # select only columns relevant for model training
        X_train = X_train.loc[:, self.feature_cols]
        logging.info(
            f"\n training data -> X: {X_train.shape}, y: {y_train.shape} \n"
            f" testing data -> X: {X_test.shape}, y: {y_test.shape}")

        logging.info("Finish preprocessing the data")
        return X_train, X_test, y_train, y_test

    @staticmethod
    def create_x_y(clean_trace: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
        """ creates two DataFrames from the clean trace that are going to be the matrices """

        logging.info("Start creating the data matrix X and the label vector y")
        # select all relevant cols for training except the label
        X = clean_trace.loc[:, clean_trace.columns != "labels"]
        y = clean_trace['labels'].to_numpy()
        y = pd.DataFrame(y, columns=["labels"], dtype='category')
        logging.info("Finish creating the data matrix X and the label vector y")
        return X, y

    def cast_and_fill_features(self, x: pd.DataFrame) -> pd.DataFrame:
        # the feature columns
        logging.info("Start casting feature columns and replacing NaN in X")
        # casting and replacing NaN
        for column in self.feature_cols:
            if column in self.bool_cols:
                x.loc[:, column] = x.loc[:, column].astype('bool')
            elif column in self.categ_cols:
                # if column in self.str_categ_cols:
                    # replace NaN with "" for string cols
                    # x.loc[:, column] = x.loc[:, column].fillna("")
                # replace NaN with the mean for numerical categories. The mean has no meaning, just a placeholder number
                # else:
                  #  x.loc[:, column] = x.loc[:, column].fillna(x.loc[:, column].mean())
                x.loc[:, column] = x.loc[:, column].astype('category')
            elif column in self.num_cols:
                # replace NaN with 0 for cols where 0 is the min
                # x.loc[:, column] = x.loc[:, column].fillna(0)
                x.loc[:, column] = x.loc[:, column].astype('float')
        logging.info("Finish casting feature columns and replacing NaN in X")
        return x

    def fit_the_data(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple([
        AutoSklearnClassifier, int]):
        logging.info("Start fitting the data")
        start_time = datetime.now()

        automl = AutoSklearnClassifier(
            time_left_for_this_task=self.fit_time,  # in seconds
            memory_limit=self.memory_limit,
            include={'feature_preprocessor': [
                'densifier',  # turn sparse vec into dense vec
                'pca',  # PCA in MLlib
                'liblinear_svc_preprocessor',  # not sure what it does but we use the model
                'polynomial',  # PolynomialExpansion in MLlib
                'random_trees_embedding',  # not sure what it does but we use the model
                'select_percentile_classification', # basic feature selection transformation
                'select_rates_classification'  # UnivariateFeatureSelector in MLlib

            ],
                'classifier': [
                    'liblinear_svc',  # LinearSVC in MLlib
                    'libsvm_svc',  # LinearSVC in MLlib
                    'gradient_boosting',  # GBTClasifier in MLlib
                    'random_forest',  # RandomForestClassifier in MLlib
                    'multinomial_nb',  # NaiveBayes in MLlib
                    'bernoulli_nb',  # NaiveBayes in MLlib'
                    'mlp'  # MultiLayerPerceptron in MLlib
                ]
            },
            scoring_functions=[balanced_accuracy, precision, recall, f1],
            per_run_time_limit=RUN_TIME_SEC,  # 1/4 of the total limit (in seconds)
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': CV_FOLDS},
            # n_jobs=2
        )

        # check if the model is already trained
        file_exists = exists(self.model_path)
        if file_exists:
            logging.info("The model already exists -> load model...")
            automl = load(self.model_path)
            results = pd.DataFrame.from_dict(automl.cv_results_)
        else:
            logging.info("The model does not exist -> train model...")
            logging.info("During fit(), models are fit on individual cross-validation folds")
            automl.fit(x_train, y_train, dataset_name=TRACE_NAME)
            logging.info("To use all available data, we call refit()"
                         " which trains all models in the final ensemble on the whole dataset.")
            automl.refit(x_train.copy(), y_train.copy())
            logging.info("persist model...")
            dump(automl, self.model_path)

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Finish fitting the data, it took {duration}")
        return automl, duration

    def predict_the_data(self, model: AutoSklearnClassifier, x_test: pd.DataFrame) -> tuple([
        pd.DataFrame, int]):

        logging.info("Start predicting the data")
        start_time = datetime.now()
        X_test_model = x_test.loc[:, self.feature_cols]
        y_predict = model.predict(X_test_model)
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Finish predicting the data, it took: {duration}")

        return y_predict, duration

    @staticmethod
    def get_eval_df(x_test: pd.DataFrame, y_test: pd.DataFrame,
                    y_predict: np.ndarray) -> pd.DataFrame:

        logging.info("Start getting an evaluation DataFrame")

        # we need a copy to not overwrite the model variables
        eval_df = x_test.copy()

        y_predict_cp = y_predict.copy()
        # in case of chunks, a df is returned
        if type(y_test) == pd.DataFrame:
            y_test_cp = y_test.to_numpy(copy=True)
        else:
            y_test_cp = y_test.copy()
        eval_df["labels"] = y_test_cp
        eval_df["prediction"] = y_predict_cp[:, ]
        logging.info("Finish getting an evaluation DataFrame")

        return eval_df

    def write_eval_df(self, eval_df: pd.DataFrame, path: str, ) -> None:
        """ writes the DF as compressed csv to {path} """
        logging.info("Start writing an evaluation DataFrame")
        # check if the output already exists
        file_exists = exists(path)
        if file_exists:
            logging.info(f"File {path} exists already")
            sys.exit()
        else:
            eval_df.to_csv(
                path_or_buf=path,
                index=False,
                compression="gzip"
            )
        logging.info("Finish writing an evaluation DataFrame")

    @staticmethod
    def plot_performance(model: AutoSklearnClassifier) -> None:
        logging.info("Start plotting the performance")

        perf_plot = model.performance_over_time_.plot(
            x='Timestamp',
            kind='line',
            legend=True,
            title='Auto-sklearn accuracy over time',
            grid=True,
        )
        fig = perf_plot.get_figure()
        plt.figure(figsize=(8, 6))
        filename = f"../../out/model/plots/auto-sklearn/performance/{FILE_NAME}.pdf"
        file_exists = exists(filename)
        if file_exists:
            logging.warning(f"plot exists already: {filename}")
        else:
            fig.savefig(filename, format="pdf")
        logging.info("Finish plotting the performance")
        return None

    @staticmethod
    def get_metric_result(cv_results):
        results = pd.DataFrame.from_dict(cv_results)
        results = results[results['status'] == "Success"]
        cols = ['rank_test_scores', 'param_classifier:__choice__', 'mean_test_score']
        cols.extend([key for key in cv_results.keys() if key.startswith('metric_')])
        return results[cols]

    def log_eval_metrics(self, y_test: pd.DataFrame, y_predict: np.ndarray,
                         model: AutoSklearnClassifier):
        logging.info("Start logging the evaluation metrics")
        y_predict_metrics = list(y_predict)
        y_test_metrics = list(y_test.to_numpy().T[0])
        assert len(y_predict_metrics) == len(y_test_metrics)
        logging.info(f"Accuracy score CV after refit:  {accuracy_score(y_test_metrics, y_predict_metrics)}")

        target_names = ["task-success", "task-failure"]
        logging.info(
            f"classification report: \n"
            f"{classification_report(y_test_metrics, y_predict_metrics, target_names=target_names)} \n"
            f"Sprint statistics: {model.sprint_statistics()} \n"
            f"Models: {model.show_models()} \n"
            f"Leaderboard: {model.leaderboard()} \n"
        )

        logging.info(self.get_metric_result(model.cv_results_).to_string(index=False))

    @staticmethod
    def mesh_plot_knn_model(eval_data: np.ndarray):
        """
        plots the binary classification results on a mesh
        :param eval_data: the dataset as a numpy array
         """
        logging.info("Starting mesh plot for knn model")
        n_neighbors = 78
        p = 1
        weights = "distance"
        h = 0.05  # step size in the mesh
        mesh_params = [str(n_neighbors), str(p), weights, str(h)]

        # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].

        # Create color maps
        cmap_light = ListedColormap(["limegreen", "lightcoral"])
        cmap_bold = ["darkgreen", "red"]

        # the two features to display in the meshplot
        feature_names = ["cpu_request", "scheduling_class"]
        assert len(feature_names) == 2

        # points in the mesh
        x_min = eval_data[:, 0].min()  # depending on the feature we might need -1 to allow negative values
        x_max = eval_data[:, 0].max()  # depending on the feature we might need +1 to increase the positive scale
        y_min = eval_data[:, 1].min()
        y_max = eval_data[:, 1].max()
        # ideally the feature values have a fixed step -> then all values can be used without a limit
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        max_records = xx.shape[0] * xx.shape[1]
        # we cant display more records than the axis limits
        eval_data = eval_data[:max_records, :]

        # Put the result into a color plot
        y_predict = eval_data[:, 3].reshape(xx.shape)  # we cant use more records than max values on an axis
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, y_predict, cmap=cmap_light)

        # Plot also the training points as circles
        sns.scatterplot(
            x=eval_data[:, 0],
            y=eval_data[:, 1],
            hue=eval_data[:, 2],  # the true label
            palette=cmap_bold,
            alpha=.6,
            edgecolor="black",
            legend="full"
        )
        # plt.legend(["success", "failure"])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(
            f"Task failure classification (k = {n_neighbors}, weights = {weights}, h = {h}, n = {max_records})"
        )

        plt.xlabel(feature_names[0])
        L = plt.legend()
        L.get_texts()[0].set_text('success')
        L.get_texts()[1].set_text('failure')
        plt.ylabel(feature_names[1])
        filename = f"../../out/model/plots/knn-model/meshplot/{FILE_NAME}-{str('-'.join(feature_names))}-{'-'.join(mesh_params)}.pdf"
        file_exists = exists(filename)
        if file_exists:
            logging.warning(f"plot exists already: {filename}")
        else:
            plt.savefig(filename, format="pdf")
        logging.info("Finishing mesh plot for knn model")

    def permut_importance_plot(self, model: AutoSklearnClassifier, x_test: pd.DataFrame,
                               y_test: pd.DataFrame):
        """ the permutation importance decrease in a model score when a given feature is randomly permuted.
        So, the higher the score, the more does the model’s predictions depend on this feature.
        https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_inspect_predictions.html#sphx-glr-examples-40-advanced-example-inspect-predictions-py
        """
        logging.info("Start permutation importance plot")
        x_test_metrics = x_test.loc[:, self.feature_cols]
        y_test_metrics = y_test.to_numpy().T[0]
        # it doesnt work
        r = permutation_importance(
            model,
            x_test_metrics,
            y_test_metrics,  # this function needs sth array-like for y
            n_repeats=10,
            random_state=SAMPLE_SEED
        )
        sort_idx = r.importances_mean.argsort()[::-1]

        plt.boxplot(r.importances[sort_idx].T,
                    labels=[self.feature_cols[i] for i in sort_idx])

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.figure(figsize=(8, 6))
        filename = f"../../plots/knn-model/permut-import-{FILE_NAME}.pdf"
        file_exists = exists(filename)
        if file_exists:
            logging.warning(f"plot exists already: {filename}")
        else:
            plt.savefig(filename, format="pdf")

        for i in sort_idx[::-1]:
            logging.info(f"{self.feature_cols[i]:10s}: {r.importances_mean[i]:.3f} +/- "
                         f"{r.importances_std[i]:.3f}")
        logging.info("Finish permutation importance plot")


if __name__ == '__main__':
    TRACE_PATH = "../../data/alibaba_clusterdata_v2018/batch_task_clean_1F_001S/part-00000-86b08754-7391-4c77-9f15-f130922adc2c-c000.csv.gz"
    CLEAN_TRACE = pd.read_csv(filepath_or_buffer=TRACE_PATH, header=0)
    logging.info(f"Pandas memory usage of the trace: {CLEAN_TRACE.memory_usage(deep=True)}")
    # all features have to be known prior to execution
    BOOL_COLS = []
    CATEG_COLS = ['task_type', 'map_reduce']
    STR_CATEG_COLS = []
    NUM_COLS = ['task_id_num', 'instance_num', 'plan_cpu', 'plan_mem', 'task_duration', 'sched_intv', 'job_exec']
    TRACE_NAME = "Alibaba2018"
    TASK_NAME = "BinaryTaskfailClassifier"
    CLEAN_TRACE_FILE_SIZE = 0.01
    MODEL_TRAIN_HOURS = 4
    FIT_TIME_SEC = int(3600 * MODEL_TRAIN_HOURS)
    RUN_TIME_SEC = int(FIT_TIME_SEC / 3)
    TEST_SIZE_FRAC = 0.2
    MEMORY_LIMIT_MB = 5000
    SAMPLE_SEED = 96
    CV_FOLDS = 5  # sklearn default

    # Important note: this app can only be run on linux machines because of auto-sklearn
    FILE_NAME = f"{TASK_NAME}-trace-{TRACE_NAME}-size-{CLEAN_TRACE_FILE_SIZE}-mem-{MEMORY_LIMIT_MB}" \
                f"-testfrac-{TEST_SIZE_FRAC}-sampleseed-{SAMPLE_SEED}-fittime-{FIT_TIME_SEC}" \
                f"-cvK-{CV_FOLDS}"
    MODEL_PATH = f'../../out/model/dump/{FILE_NAME}.joblib'
    OUTPUT_PATH = f'../../out/model/eval/{FILE_NAME}.csv.gz'

    predictor = TraceTaskFailurePredictor(
        trace=CLEAN_TRACE,
        bool_cols=BOOL_COLS,
        categ_cols=CATEG_COLS,
        str_categ_cols=STR_CATEG_COLS,
        num_cols=NUM_COLS,
        model_path=MODEL_PATH,
        test_size_frac=TEST_SIZE_FRAC,
        output_path=OUTPUT_PATH,
        fit_time=FIT_TIME_SEC,
        run_time=RUN_TIME_SEC,
        memory_limit=MEMORY_LIMIT_MB
    )
    predictor.run()
