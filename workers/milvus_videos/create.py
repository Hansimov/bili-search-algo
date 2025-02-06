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
            index_params.add_index(field_name=field_name, **field_params)
        return schema, index_params

    def create_collection(self, collection_name: str):
        schema, index_params = self.create_schema_and_indexes()
        self.milvus.client.create_collection(
            collection_name,
            schema=schema,
            index_params=index_params,
        )


if __name__ == "__main__":
    creator = MilvusCollectionCreator(verbose=True)
    creator.create_collection("videos_vecs")

    # python -m workers.milvus_videos.create
