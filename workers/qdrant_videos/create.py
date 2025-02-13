from sedb import QdrantOperator
from tclogger import logger, logstr, dict_to_str, brk

from qdrant_client.models import Distance, VectorParams

from configs.envs import QDRANT_ENVS


class QdrantCollectionCreator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.init_qdrant()

    def init_qdrant(self):
        self.qdrant = QdrantOperator(
            configs=QDRANT_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('qdrant'))}",
        )
        db_info = self.qdrant.get_db_info()
        logger.note("> Qdrant DB Info:")
        logger.mesg(dict_to_str(db_info), indent=2)

    def delete_collection(self, collection_name: str):
        db_collection_name = self.qdrant.get_db_collection_name(collection_name)
        if not self.qdrant.client.collection_exists(db_collection_name):
            logger.mesg(f"  * No existed collection: [{collection_name}]")
            return
        collection_str = logstr.file(collection_name)
        logger.warn(f"  ! WARNING: You are dropping collection: [{collection_str}]")

        confirmation = None
        while confirmation != collection_name:
            confirmation = input(f'  > Type "{collection_str}" to confirm deletion: ')
        self.qdrant.client.delete_collection(db_collection_name)
        logger.warn(f"  ✓ Dropped collection: [{collection_str}]")

    def create_collection(self, collection_name: str):
        self.delete_collection(collection_name)
        self.qdrant.client.create_collection(
            collection_name=self.qdrant.get_db_collection_name(collection_name),
            vectors_config=VectorParams(size=4, distance=Distance.DOT),
        )
        logger.success(f"  ✓ Created collection: [{logstr.file(collection_name)}]")


if __name__ == "__main__":
    creator = QdrantCollectionCreator(verbose=True)
    creator.create_collection("videos")

    # python -m workers.qdrant_videos.create
