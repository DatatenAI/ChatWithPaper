import asyncio
import os
from typing import List

from dotenv import load_dotenv

from modules.util import gen_uuid

if os.getenv('ENV') == 'DEV':
    load_dotenv()

import tiktoken
from pydantic import BaseModel
from pymilvus import connections, Collection, utility
from loguru import logger

from modules.vectors.get_embeddings import embed_text
import milvus_config.PaperDocConfig as Pdc
import milvus_config.SinglePaperConfig as Spc


def token_str(content: str):
    ENCODER = tiktoken.get_encoding("gpt2")
    return len(ENCODER.encode(content))


class PaperDocVector(BaseModel):
    doc_id: int
    pdf_hash: str
    sql_id: int
    summary_vector: List[float]


from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, IndexType


class MilvusPaperDocManager:
    def __init__(self,
                 host=os.getenv("MILVUS_HOST"),
                 port=os.getenv("MILVUS_PORT"),
                 alias="default",
                 user=os.getenv("MILVUS_USER"),
                 password=os.getenv("MILVUS_PASSWORD"),
                 collection_name=Pdc.collection_name,
                 partition_name=Pdc.partition_name,
                 schema=Pdc.schema,
                 field_name: str = Pdc.field_name,
                 index_param=Pdc.index_param,
                 output_field=Pdc.output_field,
                 nprobe=10):
        connections.connect(host=host,
                            port=port,
                            alias=alias,
                            user=user,
                            password=password)  # 连接到 Milvus 服务器

        self.collection_name = collection_name
        self.partition_name = partition_name
        self.schema = schema
        self.field_name = field_name
        self.index_param = index_param
        self.output_field = output_field
        self.nprobe = nprobe

        self.collection = self._get_collection()
        # self.collection.release()
        self.collection.load()  # 加载至内存

    def _get_collection(self):
        if utility.has_collection(self.collection_name):
            return Collection(name=self.collection_name)
        else:
            assert self.schema is not None
            assert self.index_param is not None
            assert self.field_name is not None
            try:
                collection = Collection(
                    name=self.collection_name,
                    schema=self.schema,
                    using='default',
                    shards_num=2  # 集合中分片数量
                )
                partition = collection.create_partition(partition_name=self.partition_name)
                collection.create_index(field_name=self.field_name, index_params=self.index_param)
            except Exception as e:
                print(e)
            return Collection(name=self.collection_name)

    def delete_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' deleted successfully.")
                return True
            else:
                logger.info(f"Collection '{self.collection_name}' does not exists")
                return True
        except Exception as e:
            logger.error(f"{e}")
        return False

    def insert_vectors(self, vectors, ids):
        collection = Collection(self.collection_name)
        collection.insert([self.field_name], vectors, ids)
        print("Vectors inserted successfully.")

    def search_vectors(self, query_vector: List[float], top_k: int):
        search_param = {
            "metric_type": "IP",
            "offset": 1,
            "params": {"nprobe": self.nprobe}
        }
        search_future = self.collection.search(data=[query_vector],
                                               anns_field=self.field_name,
                                               param=search_param,
                                               limit=top_k,
                                               output_fields=self.output_field,
                                               partition_names=[self.partition_name] if self.partition_name else None,
                                               _async=True)
        results = search_future.result()
        if len(results[0]) > 0:
            logger.info(f"Vectors searched, result: {[res.id for res in results[0]]}, distance:{12}")
        return results

    def update_vectors(self, vectors: List[List[float]], ids: List[int]):
        collection = Collection(self.collection_name)
        collection.update(ids, vectors, [self.field_name])
        print("Vectors updated successfully.")

class MilvusSinglePaperManager:
    def __init__(self,
                 host=os.getenv("MILVUS_HOST"),
                 port=os.getenv("MILVUS_PORT"),
                 alias="default",
                 user=os.getenv("MILVUS_USER"),
                 password=os.getenv("MILVUS_PASSWORD"),
                 collection_name=Spc.collection_name,
                 partition_name=Spc.partition_name,
                 schema=Spc.schema,
                 field_name: str = Spc.field_name,
                 index_param=Spc.index_param,
                 output_field=Spc.output_field,
                 nprobe=10):
        connections.connect(host=host,
                            port=port,
                            alias=alias,
                            user=user,
                            password=password)  # 连接到 Milvus 服务器

        self.collection_name = collection_name
        self.partition_name = partition_name
        self.schema = schema
        self.field_name = field_name
        self.index_param = index_param
        self.output_field = output_field
        self.nprobe = nprobe

        self.collection = self._get_collection()
        # self.collection.release()
        self.collection.load()  # 加载至内存

    def _get_collection(self):
        if utility.has_collection(self.collection_name):
            return Collection(name=self.collection_name)
        else:
            assert self.schema is not None
            assert self.index_param is not None
            assert self.field_name is not None
            try:
                collection = Collection(
                    name=self.collection_name,
                    schema=self.schema,
                    using='default',
                    shards_num=2  # 集合中分片数量
                )
                partition = collection.create_partition(partition_name=self.partition_name)
                collection.create_index(field_name=self.field_name, index_params=self.index_param)
            except Exception as e:
                print(e)
            return Collection(name=self.collection_name)

    def delete_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' deleted successfully.")
                return True
            else:
                logger.info(f"Collection '{self.collection_name}' does not exists")
                return True
        except Exception as e:
            logger.error(f"{e}")
        return False

    def gen_uuids(self, num: int):
        """
        产生多个uuid
        """
        ids = []
        for i in range(num):
            ids.append(gen_uuid())
        return ids
    def insert_data(self,vecs:List[List[float]], pdf_hash:str, chunk_ids:List[int],pages:List[int]):
        try:
            num_vec = len(chunk_ids)
            data = [self.gen_uuids(num_vec), vecs, [pdf_hash]*num_vec, chunk_ids, pages]
            res = self.collection.insert(data=data, partition_name=self.partition_name, _async=True)
            self.collection.flush()
        except Exception as e:
            logger.error(f"insert {self.collection_name},pdf_hash:{pdf_hash}, vec error: {e}")
            raise e

    def insert_json_data(self,json_path):
        pass

    def search_vectors(self, query_vector: List[float], top_k: int):
        search_param = {
            "metric_type": "IP",
            "offset": 1,
            "params": {"nprobe": self.nprobe}
        }
        search_future = self.collection.search(data=[query_vector],
                                               anns_field=self.field_name,
                                               param=search_param,
                                               limit=top_k,
                                               output_fields=self.output_field,
                                               partition_names=[self.partition_name] if self.partition_name else None,
                                               _async=True)
        results = search_future.result()
        if len(results[0]) > 0:
            logger.info(f"Vectors searched, result: {[res.id for res in results[0]]}")
        return results[0]

    def search_vectors_pdf(self,pdf_hash:str,query_vector: List[float], top_k: int):
        """
        从指定PDF中搜索数据
        """

        pass


    def update_vectors(self, vectors: List[List[float]], ids: List[int]):
        """
        先删除所有的pdf_hash的数据，然后再插入新的数据
        """
        pass

async def test():
    manager = MilvusSinglePaperManager(host=os.getenv("MILVUS_HOST"),
                                       port=os.getenv("MILVUS_PORT"),
                                       alias="default",
                                       user=os.getenv("MILVUS_USER"),
                                       password=os.getenv("MILVUS_PASSWORD"),
                                       collection_name=Spc.collection_name,
                                       partition_name=Spc.partition_name,
                                       schema=Spc.schema,
                                       field_name=Spc.field_name,
                                       index_param=Spc.index_param,
                                       nprobe=10)

    # 搜索向量
    query = "methods"
    query_vector = await embed_text(query)
    top_k = 10
    res = manager.search_vectors(query_vector, top_k)

if __name__ == "__main__":
    # 示例用法

    print(res)

