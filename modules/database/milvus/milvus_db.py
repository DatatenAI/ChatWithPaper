import asyncio
import os
from typing import List

from dotenv import load_dotenv

from modules.util import gen_uuid, load_data_from_json

if os.getenv('ENV') == 'DEV':
    load_dotenv()

import tiktoken
from pydantic import BaseModel
from pymilvus import connections, Collection, utility
from loguru import logger

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
                 search_field=Pdc.search_field,
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
        self.search_field = search_field
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
                collection.create_index(field_name=self.field_name, index_params=self.index_param,
                                        partition=self.partition_name)
                collection.create_index(field_name='pdf_hash', index_name="scalar_index", partition=self.partition_name)
                collection.create_index(field_name='pdf_hash', index_name="scalar_index", partition=self.partition_name)

            except Exception as e:
                logger.error(f"{repr(e)}")
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
            logger.error(f"{repr(e)}")
        return False

    def gen_uuids(self, num: int)->List[str]:
        """
        产生多个uuid
        """
        ids = []
        for i in range(num):
            ids.append(gen_uuid())
        return ids



    async def insert_data(self, ids: str, vecs: List[float],
                          pdf_hash: str, sql_id: int):
        """
        插入一条向量 ,分段 和pages id
        """
        try:
            logger.info(f"begin insert vec")
            data = [[ids], [vecs], [pdf_hash], [sql_id]]
            res = self.collection.insert(data=data, partition_name=self.partition_name, _async=True)
            self.collection.flush()
            logger.info(f"end insert {self.collection_name},pdf_hash: {pdf_hash} vector data")
        except Exception as e:
            logger.error(f"insert {self.collection_name},pdf_hash:{pdf_hash}, vec error: {repr(e)}")
            raise Exception(e)
        return res.result()

    # async def insert_json_data(self, structure_path: str):
    #     """
    #     直接把structure.json文件进行加载后上传
    #     """
    #     flat_results_json = await load_data_from_json(structure_path)
    #     vec_id = flat_results_json['vec_id']
    #     vec = flat_results_json['vectors']
    #     pdf_hash = flat_results_json['pdf_hash']
    #     sql_id = flat_results_json['sql_id']
    #     res = await self.insert_data(vec_id, vec, pdf_hash, sql_id)
    #     return res

    # ok
    def get_entity_by_sql_id(self, ids: List[int]) -> list[dict]:
        """
        输入ids然后获取对应的vectors
        """
        expr = f"sql_id in {ids}"
        logger.info(f"expr = {expr}")
        search_future = self.collection.query(
            expr=expr,
            offset=0,
            limit=10,
            output_fields=self.output_field,
            consistency_level="Strong",
            partition_names=[self.partition_name] if self.partition_name else None,
            _async=True
        )
        logger.info(f"search vec, get {len(search_future)} num data, result ids {ids}")
        return search_future

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
                                               output_fields=self.search_field,
                                               partition_names=[self.partition_name] if self.partition_name else None,
                                               _async=True)
        results = search_future.result()
        if len(results[0]) > 0:
            logger.info(f"Vectors searched, result dis: {[res.distance for res in results[0]]}")
        return results[0]

    async def search_by_vector_hash(self, pdf_hash: str, query_vector: List[float], top_k: int = 5):
        """
        从指定PDF中搜索数据
        """
        search_param = {
            "metric_type": "IP",
            "params": {"nprobe": self.nprobe}
        }
        search_future = self.collection.search(data=[query_vector],
                                               anns_field=self.field_name,
                                               param=search_param,
                                               limit=top_k,
                                               expr=f"pdf_hash == \"{pdf_hash}\"",
                                               partition_names=[self.partition_name] if self.partition_name else None,
                                               output_fields=self.search_field, _async=True)

        return search_future.result()[0]

    async def search_ids_by_pdf_hash(self, pdf_hash: str):
        """
        通过ids
        """
        result = await self.search_by_vector_hash(
            pdf_hash=pdf_hash,
            query_vector=[0.5] * 1536,
            top_k=2
        )
        ids = [res.id for res in result]
        return ids

    async def delete_by_hash(self, pdf_hash: str):
        # 搜索满足条件的向量的 ID
        try:
            ids = await self.search_ids_by_pdf_hash(pdf_hash)

            expr = f"paper_id in {ids}"
            status = self.collection.delete(expr=expr, partition_name=self.partition_name)
            logger.info(f"Data deleted successfully, pdf_hash:{pdf_hash}, {status.delete_count} datas")
        except Exception as e:
            logger.error(f"Failed to delete data. Error:{repr(e)}")


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
                 search_field=Spc.search_field,
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
        self.search_field = search_field
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
                collection.create_index(field_name=self.field_name, index_params=self.index_param, partition=self.partition_name)
                collection.create_index(field_name='pdf_hash', index_name="scalar_index", partition=self.partition_name)


            except Exception as e:
                logger.error(f"error {repr(e)}")
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
            logger.error(f"{repr(e)}")
        return False

    def gen_uuids(self, num: int):
        """
        产生多个uuid
        """
        ids = []
        for i in range(num):
            ids.append(gen_uuid())
        return ids

    async def insert_data(self,
                          ids: List[str],
                          vecs: List[List[float]],
                          pdf_hash: str,
                          chunk_ids: List[int],
                          pages: List[int],
                          sql_ids: List[int]):
        """
        插入向量 ,分段 和pages id
        """
        try:
            logger.info(f"begin insert vec")
            num_vec = len(chunk_ids)
            data = [ids, vecs, [pdf_hash] * num_vec, chunk_ids, pages, sql_ids]
            res = self.collection.insert(data=data, partition_name=self.partition_name, _async=True)
            self.collection.flush()
            logger.info(f"end insert {self.collection_name}, pdf_hash: {pdf_hash}, {num_vec} vector data")
        except Exception as e:
            logger.error(f"insert {self.collection_name},pdf_hash:{pdf_hash}, vec error: {repr(e)}")
            raise e
        return res.result()

    async def insert_json_data(self, structure_path: str, pdf_hash: str):
        """
        直接把structure.json文件进行加载后上传
        """
        flat_results_json = await load_data_from_json(structure_path)
        vecs = []
        chunk_ids = []
        pages = []
        sql_ids = []
        ids = []
        for res in flat_results_json:
            ids.append(gen_uuid())
            vecs.append(res['vectors'])
            chunk_ids.append(res['id'])
            pages.append(res['page'])
            sql_ids.append(res['sql_id'])
        res = await self.insert_data(ids, vecs, pdf_hash, chunk_ids, pages, sql_ids)
        return res

    def get_entity_by_sql_id(self, ids: List[int]) -> list[dict]:
        """
        输入ids然后获取对应的vectors
        """
        expr = f"sql_id in {ids}"
        logger.info(f"expr = {expr}")
        search_future = self.collection.query(
            expr=expr,
            offset=0,
            limit=100,
            output_fields=self.output_field,
            consistency_level="Strong",
            partition_names=[self.partition_name] if self.partition_name else None,
            _async=True
        )
        logger.info(f"search vec, get {len(search_future)} num data, result ids {ids}")
        return search_future

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
                                               output_fields=self.search_field,
                                               partition_names=[self.partition_name] if self.partition_name else None,
                                               _async=True)
        results = search_future.result()
        if len(results[0]) > 0:
            logger.info(f"Vectors searched, result dis: {[res.distance for res in results[0]]}")
        return results[0]
    async def search_by_vector_hash(self, pdf_hash: str, query_vector: List[float], top_k: int =5):
        """
        从指定PDF中搜索数据
        """
        search_param = {
            "metric_type": "IP",
            "params": {"nprobe": self.nprobe}
        }
        search_future = self.collection.search(data=[query_vector],
                                               anns_field=self.field_name,
                                               param=search_param,
                                               limit=top_k,
                                               expr=f"pdf_hash == \"{pdf_hash}\"",
                                               partition_names=[self.partition_name] if self.partition_name else None,
                                               output_fields=self.output_field, _async=True)

        return search_future.result()[0]

    async def search_ids_by_pdf_hash(self, pdf_hash: str):
        """
        通过ids
        """
        result = await self.search_by_vector_hash(
            pdf_hash=pdf_hash,
            query_vector=[0.5]*1536,
            top_k=100
        )
        ids = [res.id for res in result]
        return ids

    async def delete_by_hash(self, pdf_hash: str):
        # 搜索满足条件的向量的 ID
        try:
            ids = await self.search_ids_by_pdf_hash(pdf_hash)

            expr = f"paper_id in {ids}"
            status = self.collection.delete(expr=expr, partition_name=self.partition_name)
            logger.info(f"Data deleted successfully, pdf_hash:{pdf_hash}, {status.delete_count} datas")
        except Exception as e:
            logger.error(f"Failed to delete data. Error:{repr(e)}")


    def update_vectors(self, vectors: List[List[float]], ids: List[int]):
        """
        先删除所有的pdf_hash的数据，然后再插入新的数据
        """
        pass




async def test_paper():
    manager = MilvusPaperDocManager(host=os.getenv("MILVUS_HOST"),
                                       port=os.getenv("MILVUS_PORT"),
                                       alias="default",
                                       user=os.getenv("MILVUS_USER"),
                                       password=os.getenv("MILVUS_PASSWORD"),
                                       collection_name=Pdc.collection_name,
                                       partition_name=Pdc.partition_name,
                                       schema=Pdc.schema,
                                       field_name=Pdc.field_name,
                                       index_param=Pdc.index_param,
                                       nprobe=10)

    ids = [123]
    res =manager.get_entity_by_sql_id(ids)

    # 搜索向量
    pdf_hash = '8545e8885f13b7bfa803e71e4c1b3ac9'

    # 插入数据
    structure_path = os.path.join(os.getenv('FILE_PATH'), f"out/{pdf_hash}.json")
    res = await manager.insert_json_data(structure_path, pdf_hash)
    print(res)

async def test_single():
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
    pdf_hash = '8545e8885f13b7bfa803e71e4c1b3ac9'

    # 插入数据
    structure_path = os.path.join(os.getenv('FILE_PATH'), f"out/{pdf_hash}_structure.json")
    res = await manager.insert_json_data(structure_path, pdf_hash)

    # query = "methods"
    # query_vector, token_cost = await embed_text(query)
    # top_k = 10
    # res1 = manager.search_vectors(query_vector, top_k)

    res2 = await manager.delete_by_hash(pdf_hash=pdf_hash)
    print(res)

async def test_hash_search():

    hash = '91f04128ae2d255d7a6ebb3259f69a43'
    manager = MilvusPaperDocManager(host=os.getenv("MILVUS_HOST"),
                                    port=os.getenv("MILVUS_PORT"),
                                    alias="default",
                                    user=os.getenv("MILVUS_USER"),
                                    password=os.getenv("MILVUS_PASSWORD"),
                                    collection_name=Pdc.collection_name,
                                    partition_name=Pdc.partition_name,
                                    schema=Pdc.schema,
                                    field_name=Pdc.field_name,
                                    output_field=Pdc.output_field,
                                    index_param=Pdc.index_param,
                                    nprobe=10)

    res = await manager.search_ids_by_pdf_hash(pdf_hash=hash)
    print(res)


if __name__ == "__main__":
    # 示例用法


    # asyncio.run(test_single())

    # asyncio.run(test_paper())

    asyncio.run(test_hash_search())



