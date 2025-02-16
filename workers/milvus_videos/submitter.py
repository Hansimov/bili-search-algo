from sedb import MilvusOperator
from tclogger import logger, logstr, brk
from typing import Literal, Union

from configs.envs import MILVUS_ENVS


class MilvusVideoSubmitter:
    def __init__(self, collection: str = None, verbose: bool = False):
        self.collection = collection
        self.verbose = verbose
        self.init_milvus()

    def init_milvus(self):
        self.milvus = MilvusOperator(
            configs=MILVUS_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('milvus'))}",
        )

    def submit(
        self,
        docs: Union[dict, list[dict]],
        collection: str = None,
        op_type: Literal["insert", "upsert"] = "upsert",
    ):
        collection = collection or self.collection
        if not collection:
            logger.err(f"× Empty collection!")
            return
        try:
            if op_type == "insert":
                self.milvus.client.insert(collection_name=collection, data=docs)
            elif op_type == "upsert":
                self.milvus.client.upsert(collection_name=collection, data=docs)
            else:
                logger.err(f"× Invalid op_type: [{op_type}]")
        except Exception as e:
            logger.err(f"× Submit error: {e}")
