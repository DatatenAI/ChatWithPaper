"""
Paper筛选总结后的向量建表Field
mysql的表名和collection_name一致
"""

from pymilvus import CollectionSchema, FieldSchema, DataType


collection_name = "PaperSummaryDocVector"
partition_name = "Papers"
field_name = "summary_vector"
output_field = ["doc_id", "pdf_hash"]

paper_id = FieldSchema(
    name="doc_id",
    dtype=DataType.INT64,
    is_primary=True,
)
paper_vector = FieldSchema(
    name="summary_vector",
    dtype=DataType.FLOAT_VECTOR,
    dim=1536,
    description='vector representation of the summary'
)
paper_hash = FieldSchema(
    name="pdf_hash",
    dtype=DataType.VARCHAR,
    max_length=64,
    description='hash value of the pdf file'
)


schema = CollectionSchema(
    fields=[paper_id, paper_vector, paper_hash],
    description="Paper Vector Database"
)


index_param = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

if __name__ == "__main__":


    print(schema)

