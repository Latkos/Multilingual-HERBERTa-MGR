

class BaseModel:
    def __init__(self, model_path, model_name="bert-base-multilingual-cased"):
        self.model_path = model_path
        self.model_name = model_name

    def train(self, train_df, model_path, training_arguments, split=0.2):
        pass

    def re_evaluate_model(self,test_df, model_path=None):
        pass

    def predict(self, text, model_path=None):
        pass
