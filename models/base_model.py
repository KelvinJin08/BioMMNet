class BaseModel:
    """
    A base class for all models.
    """

    def train(self, data):
        raise NotImplementedError("Train method not implemented.")

    def predict(self, data):
        raise NotImplementedError("Predict method not implemented.")

    def save(self, model_path):
        raise NotImplementedError("Save method not implemented.")

    def load(self, model_path):
        raise NotImplementedError("Load method not implemented.")
