from sedb import ElasticOperator
from tclogger import logger, logstr, brk, dict_to_str, dict_get

from configs.envs import ELASTIC_ENVS, VIDEOS_INDEX_DEFAULT

AGG_MATCH_FILEDS = ["title.words", "tags.words", "owner.name.words", "desc.words"]
BUCKET_PATH = "aggregations.word_counts.buckets"
FILTER_PATH = [BUCKET_PATH, "took", "timed_out"]


class ElasticWordsCounter:
    def __init__(
        self,
        match_fields: list[str] = AGG_MATCH_FILEDS,
    ):
        self.es = ElasticOperator(
            ELASTIC_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('elastic'))}",
        )
        self.match_fields = match_fields

    def construct_word_query(self, word: str):
        return {
            "multi_match": {
                "query": word,
                "type": "phrase",
                "fields": self.match_fields,
            }
        }

    def construct_agg_query(self, words: list[str]):
        word_filters = {word: self.construct_word_query(word) for word in words}
        query_dict = {
            "size": 0,
            "_source": False,
            "aggs": {
                "word_counts": {"filters": {"filters": word_filters}},
            },
        }
        return query_dict

    def extract_word_counts(self, res_dict: dict):
        buckets = dict_get(res_dict, BUCKET_PATH, {})
        word_counts = {
            word: bucket.get("doc_count") for word, bucket in buckets.items()
        }
        return word_counts

    def count(self, words: list[str], sort_by_count: bool = True):
        query_dict = self.construct_agg_query(words)
        es_res = self.es.client.search(
            index=VIDEOS_INDEX_DEFAULT,
            body=query_dict,
            filter_path=FILTER_PATH,
        )
        es_res_dict = es_res.body
        word_counts = self.extract_word_counts(es_res_dict)
        if sort_by_count:
            word_counts = dict(
                sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            )
        res = {
            "word_counts": word_counts,
            **{
                field: es_res_dict.get(field)
                for field in FILTER_PATH
                if field != BUCKET_PATH
            },
        }
        return res


if __name__ == "__main__":
    words = ["我的世界", "原神", "三角洲", "王者荣耀", "玩机器", "雷军"]
    counter = ElasticWordsCounter()
    word_counts = counter.count(words)
    logger.note(f"> Counting words: {logstr.mesg(words)}")
    logger.okay(dict_to_str(word_counts, indent=2))

    # python -m models.sentencepiece.count
