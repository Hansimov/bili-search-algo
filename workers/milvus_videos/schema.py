from pymilvus import DataType

from models.fasttext.run import FasttextDocVecModelRunner

"""Useful docs:
In-memory Index | Milvus Documentation
  * https://milvus.io/docs/index.md?tab=floating
  * This doc describes suitable indexes in different scenarios.
"""

DEFAULT_SCHEMA_PARAMS = {
    "auto_id": False,
    "enable_dynamic_field": True,
}

TEXT_COLUMNS = ["title", "tags", "owner.name", "desc"]
TID_COLUMNS = ["tid", "ptid"]
OID_COLUMNS = ["owner.mid"]
TIME_COLUMNS = ["pubdate", "insert_at", "duration"]
KEEP_COLUMNS = [*TEXT_COLUMNS, *TID_COLUMNS, *OID_COLUMNS, *TIME_COLUMNS]

ARRAY_COLUMNS = ["stats"]
STATUS_COLUMNS = ["vectorized"]
DOCVEC_COLUMNS = ["title_tags_owner", "title_tags_owner_desc"]

DOCVEC_DIM = FasttextDocVecModelRunner.docvec_dim
DEFAULT_DOCVEC_SCHEMA_PARAMS = {
    "datatype": DataType.FLOAT16_VECTOR,
    "dim": DOCVEC_DIM,
}
DEFAULT_VECTOR_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {
        "M": 64,
        "efConstruction": 256,
    },
}


class MilvusVideoSchemaCreator:
    @staticmethod
    def create_bvid_schema() -> dict[str, dict]:
        return {
            "bvid": {
                "datatype": DataType.VARCHAR,
                "is_primary": True,
                "max_length": 128,
                "index_params": {
                    "index_name": "bvid_index",
                    "index_type": "Trie",
                },
            }
        }

    @staticmethod
    def create_text_schemas(columns: list = TEXT_COLUMNS) -> dict[str, dict]:
        schema = {}
        for col in columns:
            col_name = col.replace(".", "_")
            schema[col_name] = {
                "datatype": DataType.VARCHAR,
                "max_length": 4096,
                "nullable": True,
            }
            # do not index for most text columns, as we only do vector search in milvus
            if col in ["owner.name"]:
                schema[col_name]["max_length"] = 256
                schema[col_name]["index_params"] = {
                    "index_name": f"{col_name}_index",
                    "index_sort": "INVERTED",
                }
        return schema

    @staticmethod
    def create_tid_schemas(columns: list = TID_COLUMNS) -> dict[str, dict]:
        return {
            col: {
                "datatype": DataType.INT16,
                "nullable": True,
                "index_params": {
                    "index_name": f"{col}_index",
                    "index_type": "BITMAP",
                },
            }
            for col in columns
        }

    @staticmethod
    def create_oid_schemas(columns: list = OID_COLUMNS) -> dict[str, dict]:
        schema = {}
        for col in columns:
            col_name = col.replace(".", "_")
            schema[col_name] = {
                "datatype": DataType.INT64,
                "index_params": {
                    "index_name": f"{col_name}_index",
                    "index_type": "INVERTED",
                },
            }
        return schema

    @staticmethod
    def create_time_schemas(columns: list = TIME_COLUMNS) -> dict[str, dict]:
        return {
            col: {
                "datatype": DataType.INT64,
                "index_params": {
                    "index_name": f"{col}_index",
                    "index_type": "STL_SORT",
                },
            }
            for col in columns
        }

    @staticmethod
    def create_array_schemas(columns: list = ARRAY_COLUMNS) -> dict[str, dict]:
        return {
            f"{col}_arr": {
                "datatype": DataType.ARRAY,
                "element_type": DataType.INT64,
                "max_capacity": 16,
                "index_params": {
                    "index_name": f"{col}_arr_index",
                    "index_type": "AUTOINDEX",
                },
            }
            for col in columns
        }

    @staticmethod
    def create_status_schemas(columns: list = STATUS_COLUMNS) -> dict[str, dict]:
        """Flag status of related fields.
        For example, `title_status` is used to indicate whether title text is embedded as `title_vec`,
        and `tags_status` is used to indicate whether tags-related fields (such as `tags`, `title_tags_owner`, `title_tags_owner_desc`) are embedded.
        """
        return {
            f"{col}_status": {
                "datatype": DataType.INT8,
                "nullable": True,
                "index_params": {
                    "index_name": f"{col}_status_index",
                    "index_type": "BITMAP",
                },
            }
            for col in columns
        }

    @staticmethod
    def create_docvec_schemas(columns: list = DOCVEC_COLUMNS) -> dict[str, dict]:
        return {
            f"{col}_vec": {
                **DEFAULT_DOCVEC_SCHEMA_PARAMS,
                "index_params": {
                    "index_name": f"{col}_vec_index",
                    **DEFAULT_VECTOR_INDEX_PARAMS,
                },
            }
            for col in columns
        }

    @staticmethod
    def create_schemas() -> dict[str, dict]:
        return {
            **MilvusVideoSchemaCreator.create_bvid_schema(),
            **MilvusVideoSchemaCreator.create_text_schemas(),
            **MilvusVideoSchemaCreator.create_tid_schemas(),
            **MilvusVideoSchemaCreator.create_oid_schemas(),
            **MilvusVideoSchemaCreator.create_time_schemas(),
            **MilvusVideoSchemaCreator.create_array_schemas(),
            **MilvusVideoSchemaCreator.create_status_schemas(),
            **MilvusVideoSchemaCreator.create_docvec_schemas(),
        }


MILVUS_VIDEOS_COLUMNS_SCHEMA = MilvusVideoSchemaCreator.create_schemas()
