from pymilvus import DataType

from models.fasttext.run import FasttextDocVecModelRunner

"""Useful docs:
In-memory Index | Milvus Documentation
  * https://milvus.io/docs/index.md?tab=floating
  * This doc describes suitable indexes in different scenarios.
"""

DOCVEC_DIM = FasttextDocVecModelRunner.docvec_dim

DEFAULT_SCHEMA_PARAMS = {
    "auto_id": False,
    "enable_dynamic_field": True,
}

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

DOCVEC_COLUMNS = ["title", "desc", "tags", "owner", "title_tags_desc_owner"]


def create_docvec_schema_params(columns: list = DOCVEC_COLUMNS):
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


MILVUS_VIDEOS_COLUMNS_SCHEMA = {
    "bvid": {
        "datatype": DataType.VARCHAR,
        "is_primary": True,
        "max_length": 128,
        "index_params": {
            "index_type": "",
            "index_name": "bvid_index",
        },
    },
    **create_docvec_schema_params(),
}
