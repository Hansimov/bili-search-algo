# HanLP Notes

## Manually download models

Mannually download models with proxy if network is not available.

Here is an example of downloading an MTL model:

```sh
mkdir -p ~/.hanlp/mtl
HANLP_URL="https://file.hankcs.com/hanlp"
# HANLP_MODEL="mtl/en_tok_lem_pos_ner_srl_udep_sdp_con_modernbert_large_prepend_false_20250107_181612.zip"
# HANLP_MODEL="mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_base_20210111_124519.zip"
# HANLP_MODEL="transformers/xlm-roberta-base_20210706_125502.zip"
# HANLP_MODEL="mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_xlm_base_20220608_003435.zip"
HANLP_MODEL="mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_mMiniLMv2L12_no_space_20220807_133143.zip"
curl -L --proxy http://127.0.0.1:11111 -o ~/.hanlp/$HANLP_MODEL $HANLP_URL/$HANLP_MODEL
```

Resume download if interrupted:

```sh
curl -L --proxy http://127.0.0.1:11111 -o ~/.hanlp/$HANLP_MODEL -C - $HANLP_URL/$HANLP_MODEL
```

Download extra files used by HanLP:

```sh
HANLP_COM="https://file.hankcs.com"
curl -L --proxy http://127.0.0.1:11111 -o ~/.hanlp/thirdparty/file.hankcs.com/corpus/char_table.json.zip $HANLP_COM/corpus/char_table.json.zip
```
