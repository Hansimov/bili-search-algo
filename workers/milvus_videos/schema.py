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

DOCVEC_COLUMNS = [
    *["title"],
    *["tags"],
    *["title_tags_owner", "title_tags_owner_desc"],
]
ARRAY_COLUMNS = ["stats"]
STATUS_COLUMNS = ["title", "tags"]


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
                    "index_type": "AUTOINDEX",
                },
            }
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
    def create_status_schemas(columns: list = STATUS_COLUMNS) -> dict[str, dict]:
        """Flag status of related fields.
        For example, `title_status` is used to indicate whether title text is embedded as `title_vec`,
        and `tags_status` is used to indicate whether tags-related fields (such as `tags`, `title_tags_owner`, `title_tags_owner_desc`) are embedded.
        """
        return {
            f"{col}_status": {
                "datatype": DataType.INT8,
                "index_params": {
                    "index_name": f"{col}_status_index",
                    "index_type": "BITMAP",
                },
            }
            for col in columns
        }

    @staticmethod
    def create_array_schemas(columns: list = ARRAY_COLUMNS) -> dict[str, dict]:
        return {
            f"{col}_array": {
                "datatype": DataType.ARRAY,
                "element_type": DataType.INT64,
                "max_capacity": 16,
                "index_params": {
                    "index_name": f"{col}_array_index",
                    "index_type": "AUTOINDEX",
                },
            }
            for col in columns
        }

    @staticmethod
    def create_schema() -> dict[str, dict]:
        return {
            **MilvusVideoSchemaCreator.create_bvid_schema(),
            **MilvusVideoSchemaCreator.create_docvec_schemas(),
            **MilvusVideoSchemaCreator.create_status_schemas(),
            **MilvusVideoSchemaCreator.create_array_schemas(),
        }


MILVUS_VIDEOS_COLUMNS_SCHEMA = MilvusVideoSchemaCreator.create_schema()
