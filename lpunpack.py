import argparse
import copy
import enum
import io
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from string import Template
from timeit import default_timer as dti
from typing import IO, Dict, List, TypeVar, cast, BinaryIO, Tuple

# 常量定义
SPARSE_HEADER_MAGIC = 0xED26FF3A
SPARSE_HEADER_SIZE = 28
SPARSE_CHUNK_HEADER_SIZE = 12
LP_PARTITION_RESERVED_BYTES = 4096
LP_METADATA_GEOMETRY_MAGIC = 0x616C4467
LP_METADATA_GEOMETRY_SIZE = 4096
LP_METADATA_HEADER_MAGIC = 0x414C5030
LP_SECTOR_SIZE = 512
LP_TARGET_TYPE_LINEAR = 0
LP_TARGET_TYPE_ZERO = 1
LP_TARGET_TYPE_EXTEND = 2
LP_PARTITION_ATTR_READONLY = (1 << 0)
LP_PARTITION_ATTR_SLOT_SUFFIXED = (1 << 1)
LP_PARTITION_ATTR_UPDATED = (1 << 2)
LP_PARTITION_ATTR_DISABLED = (1 << 3)
LP_BLOCK_DEVICE_SLOT_SUFFIXED = (1 << 0)
LP_GROUP_SLOT_SUFFIXED = (1 << 0)

PLAIN_TEXT_TEMPLATE = """
标志: $header_flags
分区槽: $slot
元数据版本: $metadata_version
元数据大小: $metadata_size bytes
元数据最大大小: $metadata_max_size bytes
元数据槽数: $metadata_slot_count
特殊分区后缀:
a [$partition_a]
b [$partition_ab]
"""

# 枚举类型
class FormatType(enum.Enum):
    TEXT = 'text'
    JSON = 'json'

# 自定义异常
class LpUnpackError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

# 数据类定义
@dataclass
class Metadata:
    header: 'LpMetadataHeader' = field(default=None)
    geometry: 'LpMetadataGeometry' = field(default=None)
    partitions: List['LpMetadataPartition'] = field(default_factory=list)
    extents: List['LpMetadataExtent'] = field(default_factory=list)
    groups: List['LpMetadataPartitionGroup'] = field(default_factory=list)
    block_devices: List['LpMetadataBlockDevice'] = field(default_factory=list)

    @property
    def info(self) -> Dict:
        return self._get_info()

    @property
    def metadata_region(self) -> int:
        if self.geometry is None:
            return 0
        return LP_PARTITION_RESERVED_BYTES + (
                LP_METADATA_GEOMETRY_SIZE + self.geometry.metadata_max_size * self.geometry.metadata_slot_count
        ) * 2

    def _get_partition_layout(self) -> List[str]:
        result = []

        for partition in self.partitions:
            for extent_number in range(partition.num_extents):
                index = partition.first_extent_index + extent_number
                extent = self.extents[index]
                block_device_name = ""
                if extent.target_type == LP_TARGET_TYPE_LINEAR:
                    block_device_name = self.block_devices[extent.target_source].partition_name
                elif extent.target_type == LP_TARGET_TYPE_EXTEND:
                    block_device_name = self.block_devices[extent.target_source].partition_name

                result.append(
                    f"{block_device_name}: {extent.target_data} .. {extent.target_data + extent.num_sectors}: "
                    f"{partition.name} ({extent.num_sectors} sectors)"
                )

        return result

    def get_offsets(self, slot_number: int = 0) -> List[int]:
        base = LP_PARTITION_RESERVED_BYTES + (LP_METADATA_GEOMETRY_SIZE * 2)
        _tmp_offset = self.geometry.metadata_max_size * slot_number
        primary_offset = base + _tmp_offset
        backup_offset = base + self.geometry.metadata_max_size * self.geometry.metadata_slot_count + _tmp_offset
        return [primary_offset, backup_offset]

    def _get_info(self) -> Dict:
        result = {}
        try:
            partition_names = [p.name for p in self.partitions]
            has_a = any(p.endswith('_a') for p in partition_names)
            has_b = any(p.endswith('_b') for p in partition_names)

            # 判断是否存在a/b槽
            if has_a or has_b:
                slot = 'ab'
            else:
                slot = 'a'

            result = {
                "slot": slot,
                "metadata_version": f"{self.header.major_version}.{self.header.minor_version}",
                "metadata_size": self.header.header_size + self.header.tables_size,
                "metadata_max_size": self.geometry.metadata_max_size,
                "metadata_slot_count": self.geometry.metadata_slot_count,
                "header_flags": "none",
                "partition_a": "存在" if has_a else "不存在",
                "partition_ab": "存在" if has_a and has_b else "不存在"
            }
        except Exception:
            pass
        finally:
            return result

    def to_json(self) -> str:
        data = self._get_info()
        if not data:
            return ''

        return json.dumps(
            data,
            indent=1,
            cls=ShowJsonInfo,
            ignore_keys=[
                'metadata_version', 'metadata_size', 'metadata_max_size', 'metadata_slot_count', 'header_flags',
            ])

    def __str__(self):
        data = self._get_info()
        if not data:
            return ''

        template = Template(PLAIN_TEXT_TEMPLATE)
        return template.substitute(**data)

# 其他辅助函数
def build_attribute_string(attributes: int) -> str:
    attrs = []
    if attributes & LP_PARTITION_ATTR_READONLY:
        attrs.append('readonly')
    if attributes & LP_PARTITION_ATTR_SLOT_SUFFIXED:
        attrs.append('slot-suffixed')
    if attributes & LP_PARTITION_ATTR_UPDATED:
        attrs.append('updated')
    if attributes & LP_PARTITION_ATTR_DISABLED:
        attrs.append('disabled')
    return ','.join(attrs) if attrs else 'none'

def build_block_device_flag_string(flags: int) -> str:
    return "slot-suffixed" if (flags & LP_BLOCK_DEVICE_SLOT_SUFFIXED) else 'none'

def build_group_flag_string(flags: int) -> str:
    return "slot-suffixed" if (flags & LP_GROUP_SLOT_SUFFIXED) else "none"

class ShowJsonInfo(json.JSONEncoder):
    def __init__(self, ignore_keys: List[str], **kwargs):
        super().__init__(**kwargs)
        self._ignore_keys = ignore_keys

    def _remove_ignore_keys(self, data: Dict):
        _data = copy.deepcopy(data)
        for field_key, v in data.items():
            if field_key in self._ignore_keys:
                _data.pop(field_key)
                continue

            if v == 0:
                _data.pop(field_key)
                continue

            if isinstance(v, int) and not isinstance(v, bool):
                _data.update({field_key: str(v)})
        return _data

    def encode(self, data: Dict) -> str:
        result = self._remove_ignore_keys(data)
        return super().encode(result)

# 元数据结构定义
class SparseHeader:
    def __init__(self, buffer):
        fmt = '<I4H4I'
        (
            self.magic,
            self.major_version,
            self.minor_version,
            self.file_hdr_sz,
            self.chunk_hdr_sz,
            self.blk_sz,
            self.total_blks,
            self.total_chunks,
            self.image_checksum
        ) = struct.unpack(fmt, buffer[0:struct.calcsize(fmt)])

class SparseChunkHeader:
    def __init__(self, buffer):
        fmt = '<2H2I'
        (
            self.chunk_type,
            self.reserved,
            self.chunk_sz,
            self.total_sz
        ) = struct.unpack(fmt, buffer[0:struct.calcsize(fmt)])

class LpMetadataBase:
    _fmt = None

    @classmethod
    @property
    def size(cls) -> int:
        return struct.calcsize(cls._fmt)

class LpMetadataGeometry(LpMetadataBase):
    _fmt = '<2I32s3I'

    def __init__(self, buffer):
        (
            self.magic,
            self.struct_size,
            self.checksum,
            self.metadata_max_size,
            self.metadata_slot_count,
            self.logical_block_size
        ) = struct.unpack(self._fmt, buffer[0:struct.calcsize(self._fmt)])

class LpMetadataTableDescriptor(LpMetadataBase):
    _fmt = '<3I'

    def __init__(self, buffer):
        (
            self.offset,
            self.num_entries,
            self.entry_size
        ) = struct.unpack(self._fmt, buffer[:struct.calcsize(self._fmt)])

class LpMetadataPartition(LpMetadataBase):
    _fmt = '<36s4I'

    def __init__(self, buffer):
        (
            self.name,
            self.attributes,
            self.first_extent_index,
            self.num_extents,
            self.group_index
        ) = struct.unpack(self._fmt, buffer[0:struct.calcsize(self._fmt)])
        self.name = self.name.decode("utf-8").strip('\x00')

    @property
    def filename(self) -> str:
        return f'{self.name}.img'

class LpMetadataExtent(LpMetadataBase):
    _fmt = '<QIQI'

    def __init__(self, buffer):
        (
            self.num_sectors,
            self.target_type,
            self.target_data,
            self.target_source
        ) = struct.unpack(self._fmt, buffer[0:struct.calcsize(self._fmt)])

class LpMetadataHeader(LpMetadataBase):
    _fmt = '<I2hI32sI32s'

    partitions: 'LpMetadataTableDescriptor' = field(default=None)
    extents: 'LpMetadataTableDescriptor' = field(default=None)
    groups: 'LpMetadataTableDescriptor' = field(default=None)
    block_devices: 'LpMetadataTableDescriptor' = field(default=None)

    def __init__(self, buffer):
        (
            self.magic,
            self.major_version,
            self.minor_version,
            self.header_size,
            self.header_checksum,
            self.tables_size,
            self.tables_checksum
        ) = struct.unpack(self._fmt, buffer[0:struct.calcsize(self._fmt)])
        self.flags = 0

class LpMetadataPartitionGroup(LpMetadataBase):
    _fmt = '<36sIQ'

    def __init__(self, buffer):
        (
            self.name,
            self.flags,
            self.maximum_size
        ) = struct.unpack(self._fmt, buffer[0:struct.calcsize(self._fmt)])

        self.name = self.name.decode("utf-8").strip('\x00')

class LpMetadataBlockDevice(LpMetadataBase):
    _fmt = '<Q2IQ36sI'

    def __init__(self, buffer):
        (
            self.first_logical_sector,
            self.alignment,
            self.alignment_offset,
            self.block_device_size,
            self.partition_name,
            self.flags
        ) = struct.unpack(self._fmt, buffer[0:struct.calcsize(self._fmt)])

        self.partition_name = self.partition_name.decode("utf-8").strip('\x00')

@dataclass
class UnpackJob:
    name: str
    geometry: 'LpMetadataGeometry'
    parts: List[Tuple[int, int]] = field(default_factory=list)
    total_size: int = field(default=0)

# 主处理类
class LpUnpack:
    def __init__(self, **kwargs):
        self._partition_name = kwargs.get('NAME')
        self._show_info = kwargs.get('SHOW_INFO', True)
        self._show_info_format = kwargs.get('SHOW_INFO_FORMAT', FormatType.TEXT)
        self._config = kwargs.get('CONFIG', None)
        self._slot_num = kwargs.get('SLOT_NUM', None)
        self._fd: BinaryIO = open(kwargs.get('SUPER_IMAGE'), 'rb')
        self._out_dir = kwargs.get('OUTPUT_DIR', None)

    def _check_out_dir_exists(self):
        if self._out_dir is None:
            return
        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir, exist_ok=True)

    def _extract_partition(self, unpack_job: UnpackJob):
        self._check_out_dir_exists()
        start = dti()
        print(f'- 正在提取 {unpack_job.name}')
        out_file = os.path.join(self._out_dir, f'{unpack_job.name}.img')
        with open(str(out_file), 'wb') as out:
            for part in unpack_job.parts:
                offset, size = part
                self._write_extent_to_file(out, offset, size, unpack_job.geometry.logical_block_size)
        print(f'- 完成！耗时 [{dti() - start:.2f} 秒]')

    def _extract(self, partition, metadata):
        unpack_job = UnpackJob(name=partition.name, geometry=metadata.geometry)

        if partition.num_extents != 0:
            for extent_number in range(partition.num_extents):
                index = partition.first_extent_index + extent_number
                extent = metadata.extents[index]

                if extent.target_type not in (LP_TARGET_TYPE_LINEAR, LP_TARGET_TYPE_EXTEND):
                    raise LpUnpackError(f'不支持的扩展类型: {extent.target_type}')

                offset = extent.target_data * LP_SECTOR_SIZE
                size = extent.num_sectors * LP_SECTOR_SIZE
                unpack_job.parts.append((offset, size))
                unpack_job.total_size += size

        self._extract_partition(unpack_job)

    def _get_data(self, count: int, size: int, clazz: TypeVar('T')) -> List[TypeVar('T')]:
        result = []
        while count > 0:
            data = self._fd.read(size)
            if not data or len(data) < size:
                raise LpUnpackError('读取数据时遇到意外的文件结束。')
            result.append(clazz(data))
            count -= 1
        return result

    def _read_metadata_header(self, metadata: Metadata):
        offsets = metadata.get_offsets()
        for index, offset in enumerate(offsets):
            self._fd.seek(offset, io.SEEK_SET)
            header = LpMetadataHeader(self._fd.read(struct.calcsize(LpMetadataHeader._fmt)))
            header.partitions = LpMetadataTableDescriptor(self._fd.read(struct.calcsize(LpMetadataTableDescriptor._fmt)))
            header.extents = LpMetadataTableDescriptor(self._fd.read(struct.calcsize(LpMetadataTableDescriptor._fmt)))
            header.groups = LpMetadataTableDescriptor(self._fd.read(struct.calcsize(LpMetadataTableDescriptor._fmt)))
            header.block_devices = LpMetadataTableDescriptor(self._fd.read(struct.calcsize(LpMetadataTableDescriptor._fmt)))

            if header.magic != LP_METADATA_HEADER_MAGIC:
                check_index = index + 1
                if check_index >= len(offsets):
                    raise LpUnpackError('逻辑分区元数据的魔数无效。')
                else:
                    print(f'通过偏移量 0x{offsets[check_index]:x} 读取备份头')
                    continue

            metadata.header = header
            self._fd.seek(offset + header.header_size, io.SEEK_SET)

    def _read_metadata(self):
        self._fd.seek(LP_PARTITION_RESERVED_BYTES, io.SEEK_SET)
        metadata = Metadata(geometry=self._read_primary_geometry())

        if metadata.geometry.magic != LP_METADATA_GEOMETRY_MAGIC:
            raise LpUnpackError('逻辑分区元数据的几何魔数无效。')

        if metadata.geometry.metadata_slot_count == 0:
            raise LpUnpackError('逻辑分区元数据的槽位计数无效。')

        if metadata.geometry.metadata_max_size % LP_SECTOR_SIZE != 0:
            raise LpUnpackError('元数据最大大小未对齐到扇区。')

        self._read_metadata_header(metadata)

        metadata.partitions = self._get_data(
            metadata.header.partitions.num_entries,
            metadata.header.partitions.entry_size,
            LpMetadataPartition
        )

        metadata.extents = self._get_data(
            metadata.header.extents.num_entries,
            metadata.header.extents.entry_size,
            LpMetadataExtent
        )

        metadata.groups = self._get_data(
            metadata.header.groups.num_entries,
            metadata.header.groups.entry_size,
            LpMetadataPartitionGroup
        )

        metadata.block_devices = self._get_data(
            metadata.header.block_devices.num_entries,
            metadata.header.block_devices.entry_size,
            LpMetadataBlockDevice
        )

        try:
            super_device: LpMetadataBlockDevice = cast(LpMetadataBlockDevice, iter(metadata.block_devices).__next__())
            if metadata.metadata_region > super_device.first_logical_sector * LP_SECTOR_SIZE:
                raise LpUnpackError('逻辑分区元数据与逻辑分区内容重叠。')
        except StopIteration:
            raise LpUnpackError('元数据未指定超级设备。')

        return metadata

    def _read_primary_geometry(self) -> 'LpMetadataGeometry':
        geometry = LpMetadataGeometry(self._fd.read(LP_METADATA_GEOMETRY_SIZE))
        if geometry is not None:
            return geometry
        else:
            return LpMetadataGeometry(self._fd.read(LP_METADATA_GEOMETRY_SIZE))

    def _write_extent_to_file(self, fd: IO, offset: int, size: int, block_size: int):
        self._fd.seek(offset)
        remaining = size
        while remaining > 0:
            read_size = min(block_size, remaining)
            block = self._fd.read(read_size)
            if not block:
                raise LpUnpackError('读取分区数据时遇到意外的文件结束。')
            fd.write(block)
            remaining -= len(block)

    def get_info(self):
        try:
            self._fd.seek(0)
            metadata = self._read_metadata()

            filter_partition = []
            for partition in metadata.partitions:
                if not self._partition_name or partition.name in self._partition_name:
                    filter_partition.append(partition.name)

            if not filter_partition:
                raise LpUnpackError(f'找不到指定的分区: {self._partition_name}')

            return filter_partition

        except LpUnpackError as e:
            print(e.message)
            sys.exit(1)

        finally:
            self._fd.close()

    def list_partitions(self):
        try:
            self._fd.seek(0)
            metadata = self._read_metadata()

            partition_names = [partition.name for partition in metadata.partitions]

            if not partition_names:
                raise LpUnpackError('未找到任何分区。')

            for name in partition_names:
                print(name)

        except LpUnpackError as e:
            print(e.message)
            sys.exit(1)

        finally:
            self._fd.close()

    def unpack(self):
        try:
            self._fd.seek(0)
            metadata = self._read_metadata()

            if self._partition_name:
                filter_partition = []
                for partition in metadata.partitions:
                    if partition.name in self._partition_name:
                        filter_partition.append(partition)

                if not filter_partition:
                    raise LpUnpackError(f'找不到指定的分区: {self._partition_name}')

                metadata.partitions = filter_partition

            if self._slot_num:
                if self._slot_num > metadata.geometry.metadata_slot_count:
                    raise LpUnpackError(f'无效的元数据槽位编号: {self._slot_num}')

            if self._show_info:
                if self._show_info_format == FormatType.TEXT:
                    print(metadata)
                elif self._show_info_format == FormatType.JSON:
                    print(f"{metadata.to_json()}\n")

            if not self._show_info and self._out_dir is None:
                raise LpUnpackError(message='未指定提取目录')

            if self._out_dir:
                for partition in metadata.partitions:
                    self._extract(partition, metadata)

        except LpUnpackError as e:
            print(e.message)
            sys.exit(1)

        finally:
            self._fd.close()

def main():
    print("Android15 Super镜像提取器")
    print("By Kamenta")
    parser = argparse.ArgumentParser(description='Android15 Super Image Unpacker')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true', help='仅列出super.img中的分区名称')
    group.add_argument('--unpack', action='store_true', help='解包指定分区到输出目录')

    parser.add_argument('--super', required=True, help='路径到super.img文件')
    parser.add_argument('--out', help='提取输出目录')
    parser.add_argument('--partitions', nargs='*', help='指定要提取的分区名称')
    parser.add_argument('--slot_num', type=int, help='指定元数据槽位编号')
    parser.add_argument('--format', choices=[e.value for e in FormatType], default='text', help='输出格式')

    args = parser.parse_args()

    if not os.path.exists(args.super):
        print(f"{args.super} 未找到")
        sys.exit(1)

    namespace = argparse.Namespace(
        SUPER_IMAGE=args.super,
        OUTPUT_DIR=args.out,
        SHOW_INFO=args.unpack,
        SHOW_INFO_FORMAT=FormatType(args.format),
        NAME=args.partitions,
        SLOT_NUM=args.slot_num
    )

    lpunpack = LpUnpack(**vars(namespace))

    if args.list:
        lpunpack.list_partitions()
    elif args.unpack:
        if not args.out:
            print("解包必须指定输出目录 --out")
            sys.exit(1)
        lpunpack.unpack()

def get_parts(file_: str):
    try:
        namespace = argparse.Namespace(
            SUPER_IMAGE=file_,
            SHOW_INFO=False,
            NAME=None,
            SLOT_NUM=None,
            OUTPUT_DIR=None
        )
        return LpUnpack(**vars(namespace)).get_info()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_} 未找到")

if __name__ == "__main__":
    main()
