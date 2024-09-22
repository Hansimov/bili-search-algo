from sedb import MongoOperator
from configs.envs import MONGO_ENVS


class DataLoader:
    def __init__(self):
        self.mongo = MongoOperator(
            MONGO_ENVS, connect_msg=f"from {self.__class__.__name__}"
        )
    
    def next(self, batch_size: int = 10):
        # return self.mongo.get_batch(batch_size)
        pass


if __name__ == "__main__":
    loader = DataLoader()

    # python -m models.data_loader
