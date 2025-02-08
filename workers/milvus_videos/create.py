from pymilvus import MilvusClient, CollectionSchema
from pymilvus.milvus_client.index import IndexParams
from sedb import MilvusOperator
from tclogger import logger, logstr, dict_to_str, brk

from configs.envs import MILVUS_ENVS
from workers.milvus_videos.schema import DEFAULT_SCHEMA_PARAMS
from workers.milvus_videos.schema import MILVUS_VIDEOS_COLUMNS_SCHEMA


class MilvusCollectionCreator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.init_milvus()

    def init_milvus(self):
        self.milvus = MilvusOperator(
            configs=MILVUS_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('milvus'))}",
        )
        db_info = self.milvus.get_db_info()
        logger.note("> Milvus DB Info:")
        logger.mesg(dict_to_str(db_info), indent=2)

    def create_schema_and_indexes(
        self, columns_schema: dict[str, dict] = MILVUS_VIDEOS_COLUMNS_SCHEMA
    ) -> tuple[CollectionSchema, IndexParams]:
        logger.note("> Creating schema and indexes:")
        logger.mesg(dict_to_str(columns_schema), indent=2)
        # init schema
        schema = MilvusClient.create_schema(**DEFAULT_SCHEMA_PARAMS)
        # pop index_params from columns info
        index_params_dict = {
            k: v.pop("index_params", {}) for k, v in columns_schema.items()
        }
        # add fields
        for field_name, field_params in columns_schema.items():
            schema.add_field(field_name=field_name, **field_params)
        # add indexes
        index_params = self.milvus.client.prepare_index_params()
        for field_name, field_params in index_params_dict.items():
            if field_params:
                index_params.add_index(field_name=field_name, **field_params)
        return schema, index_params

    def drop_collection(self, collection_name: str):
        if not self.milvus.client.has_collection(collection_name):
            logger.mesg(f"  * No existed collection: [{collection_name}]")
            return
        collection_str = logstr.file(collection_name)
        logger.warn(f"  ! WARNING: You are dropping collection: [{collection_str}]")

        confirmation = None
        while confirmation != collection_name:
            confirmation = input(f'  > Type "{collection_str}" to confirm deletion: ')
        self.milvus.client.drop_collection(collection_name)
        logger.warn(f"  ✓ Dropped collection: [{collection_str}]")

    def create_collection(self, collection_name: str):
        schema, index_params = self.create_schema_and_indexes()
        self.drop_collection(collection_name)
        self.milvus.client.create_collection(
            collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.success(f"  ✓ Created collection: [{logstr.file(collection_name)}]")


if __name__ == "__main__":
    creator = MilvusCollectionCreator(verbose=True)
    creator.create_collection("videos")

    # python -m workers.milvus_videos.create
