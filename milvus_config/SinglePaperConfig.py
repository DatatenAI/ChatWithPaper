"""
Paper筛选总结后的向量建表Field
mysql的表名和collection_name一致
"""

from pymilvus import CollectionSchema, FieldSchema, DataType

collection_name = "SinglePaperDocVector"
partition_name = "SinglePapers"
field_name = "chunk_vector"
output_field = ["paper_id", "pdf_hash", "chunk_id", "page"]

paper_id = FieldSchema(
    name="paper_id",
    dtype=DataType.VARCHAR,
    max_length=64,
    is_primary=True,
)

chunk_vector = FieldSchema(
    name=field_name,
    dtype=DataType.FLOAT_VECTOR,
    dim=1536,
    description='chunk vector representation of the chunk text'
)

paper_hash = FieldSchema(
    name="pdf_hash",
    dtype=DataType.VARCHAR,
    max_length=64,
    description='hash value of the pdf file'
)

chunk_id = FieldSchema(
    name="chunk_id",
    dtype=DataType.INT64,
    description="chunk id of the json file"
)

page = FieldSchema(
    name="page",
    dtype=DataType.INT64,
    description="page number in the files"
)

sql_id = FieldSchema(
    name="sql_id",
    dtype=DataType.INT64,
    description='sql id for paper_chunks'
)

schema = CollectionSchema(
    fields=[paper_id, chunk_vector, paper_hash, chunk_id, page, sql_id],
    description="Paper Vector Database"
)

index_param = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}


async def test_insert_data(coll):
    from modules.util import load_data_from_json
    import os
    from dotenv import load_dotenv
    from loguru import logger
    load_dotenv()
    from modules.util import gen_uuid

    pdf_hash = '8545e8885f13b7bfa803e71e4c1b3ac9'
    structure_path = os.path.join(os.getenv('FILE_PATH'), f"out/{pdf_hash}_structure.json")
    logger.info("begin load json")
    flat_results_json = await load_data_from_json(structure_path)
    logger.info(f"load success")
    print(flat_results_json[0])
    ids = []
    vecs = []
    hashs = []
    chunk_ids = []
    pages = []
    sql_id = []
    flag = 0
    for res in flat_results_json:
        ids.append(gen_uuid())
        vecs.append(res['vectors'])
        chunk_ids.append(res['id'])
        hashs.append(pdf_hash)
        pages.append(res['page'])
        sql_id.append(flag)
        flag += 1
    insert_data = [ids, vecs, hashs, chunk_ids, pages, sql_id]
    res = coll.insert(data=insert_data, partition_name=partition_name, _async=True)
    coll.flush()
    print(res)


if __name__ == "__main__":
    from pymilvus import connections
    from pymilvus import utility
    from pymilvus import utility, Collection
    from pymilvus import CollectionSchema, FieldSchema, DataType
    import asyncio

    connections.connect(
        alias='default',
        user='root',
        # password='MilvusChatPaper@123',
        password='test',
        host='121.37.21.153',
        port='19530'
    )

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    # print(f"collection_name: {collection_name},{utility.has_collection(collection_name)}")
    #
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2  # 集合中分片数量
    )
    partition = collection.create_partition(partition_name=partition_name)
    #
    # print(collection.partitions)
    #
    collection.create_index(field_name=field_name, index_params=index_param)
    # print(collection.index().params)
    collection.load()

    asyncio.run(test_insert_data(collection))
    # test insert data
