class BaseModel:
    def __init__(self, model_path, model_type="bert-base-multilingual-cased"):
        self.model_path = model_path
        self.model_type = model_type

    def train(self, train_df, training_arguments=None, split=0.2):
        pass

    def re_evaluate_model(self, test_df, model_path=None):
        pass

    def predict(self, sentences, model_path):
        pass
