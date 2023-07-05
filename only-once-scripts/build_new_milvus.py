from pymilvus import connections
from pymilvus import utility, Collection
import milvus_config.SinglePaperConfig as Spc
import milvus_config.PaperDocConfig as Pdc
from loguru import logger

host_ip = '121.37.21.153'

# single
connections.connect(
    alias='default',
    user='root',
    # password='MilvusChatPaper@123',
    password='test',
    host=host_ip,
    port='19530'
)

if utility.has_collection(Spc.collection_name):
    logger.info(f"Has collection:{Spc.collection_name}")
    utility.drop_collection(Spc.collection_name)
    logger.info(f"Drop collection:{Spc.collection_name}")
# print(f"collection_name: {collection_name},{utility.has_collection(collection_name)}")
#
collection = Collection(
    name=Spc.collection_name,
    schema=Spc.schema,
    using='default',
    shards_num=2  # 集合中分片数量
)
partition = collection.create_partition(partition_name=Spc.partition_name)
#
# print(collection.partitions)
#
collection.create_index(field_name=Spc.field_name, index_params=Spc.index_param)
# print(collection.index().params)
collection.load()
logger.info(f"Create collection:{Spc.collection_name}")

#   PaperDoc

if utility.has_collection(Pdc.collection_name):
    logger.info(f"Has collection:{Pdc.collection_name}")
    utility.drop_collection(Pdc.collection_name)
    logger.info(f"Drop collection:{Pdc.collection_name}")
# print(f"collection_name: {collection_name},{utility.has_collection(collection_name)}")
#
collection = Collection(
    name=Pdc.collection_name,
    schema=Pdc.schema,
    using='default',
    shards_num=2  # 集合中分片数量
)
partition = collection.create_partition(partition_name=Pdc.partition_name)
#
# print(collection.partitions)
#
collection.create_index(field_name=Pdc.field_name, index_params=Pdc.index_param)
# print(collection.index().params)
collection.load()
logger.info(f"Create collection:{Pdc.collection_name}")